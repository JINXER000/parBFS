# include "ubfsfunc.h"
# include "uiucbfs.cuh"
# include "timer.h"
// int3: next_wf, local-Q
namespace ubfs
{

void parWavefront(std::vector<int> &srcNode,
		Graph<AdjacentNode> &cuGraph,
		std::vector<int> &distances,
		std::vector<int> &cameFrom,
		int destination)
{


	thrust::device_vector<int> rounds(1);
	rounds[0]=0;
	thrust::device_vector<int> d_cameFrom(cuGraph.numVertices);
	thrust::copy(cameFrom.begin(),cameFrom.end(),d_cameFrom.begin());

	thrust::device_vector<Edge> d_graph_node(cuGraph.numVertices);
	for(int i=0;i<cuGraph.numVertices;i++)
	{
		int2 tmp;
		tmp.x=cuGraph.edgesOffset[i];
		tmp.y=cuGraph.edgesSize[i];
		d_graph_node[i]=tmp;
	}
	thrust::device_vector<Node> d_graph_edge(cuGraph.numEdges);
	for(int i=0;i<cuGraph.numEdges;i++)
	{
		int2 tmp;
		tmp.x=cuGraph.adjacencyList[i].terminalVertex;
		tmp.y=cuGraph.adjacencyList[i].weight;
		d_graph_edge[i]=tmp;
	}

	ubfsGraph<int> ugraph(cuGraph.numEdges,cuGraph.numVertices);
	thrust::copy(distances.begin(),distances.end(),ugraph.cost_shared.begin());
	ugraph.q1_shared[0]=srcNode[0];
	int * d_q1= raw_pointer_cast(&ugraph.q1_shared[0]);
	int * d_q2= raw_pointer_cast(&ugraph.q2_shared[0]);





	//whether or not to adjust "k", see comment on "BFS_kernel_multi_blk_inGPU" for more details
	int * num_td;//number of threads
	cudaMalloc((void**) &num_td, sizeof(int));


	int num_t;//number of threads
	int k=0;//BFS level index
	int num_of_blocks;
	int num_of_threads_per_block;


	GpuTimer tm1;
	tm1.Start();
	do
	{
		num_t=ugraph.tail_shared[0];
		ugraph.tail_shared[0]=0;

		if(num_t == 0){//frontier is empty
			cudaFree(num_td);
			break;
		}

		num_of_blocks = 1;
		num_of_threads_per_block = num_t;
		if(num_of_threads_per_block <NUM_BIN)
			num_of_threads_per_block = NUM_BIN;
		if(num_t>MAX_THREADS_PER_BLOCK)
		{
			num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK);
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
		}
		if(num_of_blocks == 1)//will call "BFS_in_GPU_kernel"
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
		if(num_of_blocks >1 && num_of_blocks <= NUM_SM)// will call "BFS_kernel_multi_blk_inGPU"
			num_of_blocks = NUM_SM;

		//assume "num_of_blocks" can not be very large
		dim3  grid( num_of_blocks, 1, 1);
		dim3  threads( num_of_threads_per_block, 1, 1);

		if(k%2 == 0){
			if(num_of_blocks == 1){
				BFS_in_GPU_kernel<int><<< grid, threads >>>(ugraph,d_q1,d_q2, raw_pointer_cast(&d_graph_node[0]),
						raw_pointer_cast(&d_graph_edge[0]), num_t , GRAY0,k,
						destination, raw_pointer_cast(&rounds[0]),raw_pointer_cast(&d_cameFrom[0]));
			}
			else if(num_of_blocks <= NUM_SM){
				(cudaMemcpy(num_td,&num_t,sizeof(int),
						cudaMemcpyHostToDevice));
				BFS_kernel_multi_blk_inGPU<int>
				<<< grid, threads >>>(ugraph,d_q1,d_q2, raw_pointer_cast(&d_graph_node[0]),
						raw_pointer_cast(&d_graph_edge[0]),  num_td, GRAY0,k,
						destination,raw_pointer_cast(&d_cameFrom[0]));
				int switch_k= ugraph.switchk_shared[0];
				if(!switch_k){
					k--;
				}
			}
			else{
				BFS_kernel<int><<< grid, threads >>>(ugraph,d_q1,d_q2, raw_pointer_cast(&d_graph_node[0]),
						raw_pointer_cast(&d_graph_edge[0]),  num_t, GRAY0,k,
						destination,raw_pointer_cast(&d_cameFrom[0]));
			}
		}
		else{
			if(num_of_blocks == 1){
				BFS_in_GPU_kernel<int><<< grid, threads >>>(ugraph,d_q2,d_q1, raw_pointer_cast(&d_graph_node[0]),
						raw_pointer_cast(&d_graph_edge[0]),  num_t, GRAY1,k,
						destination,raw_pointer_cast(&rounds[0]),raw_pointer_cast(&d_cameFrom[0]));
			}
			else if(num_of_blocks <= NUM_SM){
				(cudaMemcpy(num_td,&num_t,sizeof(int),
						cudaMemcpyHostToDevice));
				BFS_kernel_multi_blk_inGPU<int>
				<<< grid, threads >>>(ugraph,d_q2,d_q1, raw_pointer_cast(&d_graph_node[0]),
						raw_pointer_cast(&d_graph_edge[0]), num_td, GRAY1,k,
						destination,raw_pointer_cast(&d_cameFrom[0]));
				int switch_k= ugraph.switchk_shared[0];
				if(!switch_k){
					k--;
				}
			}
			else{
				BFS_kernel<int><<< grid, threads >>>(ugraph,d_q2,d_q1, raw_pointer_cast(&d_graph_node[0]),
						raw_pointer_cast(&d_graph_edge[0]),  num_t,  GRAY1,k,
						destination,raw_pointer_cast(&d_cameFrom[0]));
			}
		}
		k++;
		int h_overflow= ugraph.overflow_shared[0];
		if(h_overflow) {
			printf("Error: local queue was overflow. Need to increase W_LOCAL_QUEUE\n");
			return;
		}
		// copy is end d 2 h
		int h_is_end= ugraph.isend_shared[0];
		if(h_is_end)
		{
			break;
		}
	} while(1);
	tm1.Stop();
	std::cout<<"total time is "<<float(tm1.Elapsed())<<" ms"<<std::endl;
	cudaThreadSynchronize();
	printf("GPU kernel done\n");

	// copy result from device to host
	// copy dist d 2 h
	thrust::copy(ugraph.cost_shared.begin(),ugraph.cost_shared.end(),distances.begin());
	int rnd=rounds[0];
	printf("final distance is %d , levels  is %d \n",distances[destination],rnd);
	// retrace
	//		std::vector<int> search_path;
	//		int ptOnPath=destination;
	//		while(d_cameFrom[ptOnPath]!= ptOnPath)
	//		{
	//			search_path.push_back(ptOnPath);
	//			ptOnPath= d_cameFrom[ptOnPath];
	//		}
	//	//	std::reverse(search_path.begin(),search_path.end());
	//		for(auto ptr:search_path)
	//		{
	//			std::cout<<ptr<<",";
	//		}



}

}
