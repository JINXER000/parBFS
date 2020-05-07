#pragma once

#include <cuda_utils.h>
#include "ubfs_config.h"

namespace ubfs
{


typedef int2 Node;
typedef int2 Edge;

//texture<Node> g_graph_node_ref;
//texture<Edge> g_graph_edge_ref;

template <class Ktype>
struct ubfsGraphBase
{
	// global barrier
	int count;
	int no_of_nodes_vol ;
	int stay_vol;

	// param
	int numEdges,numVertices;
	int * tail;
	int * switchk;
	int *global_kt;
	int *overflow;
	bool *isend;

	// graph
	int* colorD;
	int* costD;
	Ktype *q1D;
	Ktype *q2D;




	ubfsGraphBase(int ne,int nv):numEdges(ne),numVertices(nv)
{
		reset_barrier();
}

	void reset_barrier()
	{
		 count = 0;
		 no_of_nodes_vol = 0;
		 stay_vol = 0;
	}

};

template <class Ktype, class memspace=device_memspace>
class ubfsGraph: public ubfsGraphBase<Ktype>
{
public:
	// arr
	typedef ubfsGraphBase<Ktype> parent_type;
	typename vector_type<int,memspace>::type color_shared;
	typename vector_type<int,memspace>::type cost_shared;
	typename vector_type<Ktype,memspace>::type q1_shared;
	typename vector_type<Ktype,memspace>::type q2_shared;
	typename vector_type<int,memspace>::type tail_shared;
	typename vector_type<int,memspace>::type switchk_shared;
	typename vector_type<int,memspace>::type global_kt_shared;
	typename vector_type<int,memspace>::type overflow_shared;
	typename vector_type<bool,memspace>::type isend_shared;

	ubfsGraph(int ne,int nv):parent_type(ne,nv),
			color_shared(nv), cost_shared(nv),
			q1_shared(nv), q2_shared(nv),
			tail_shared(1), switchk_shared(1),
			global_kt_shared(1),
			overflow_shared(1), isend_shared(1)
	{
		tail_shared[0]= 1;
		global_kt_shared[0]=0;
		overflow_shared[0]= 0;
		isend_shared[0]= false;
		thrust::fill(color_shared.begin(),color_shared.end(),WHITE);

		using thrust::raw_pointer_cast;
		this->colorD= raw_pointer_cast(&color_shared[0]);
		this->costD= raw_pointer_cast(&cost_shared[0]);
		this->q1D= raw_pointer_cast(&q1_shared[0]);
		this->q2D= raw_pointer_cast(&q2_shared[0]);
		this->tail = raw_pointer_cast(&tail_shared[0]);
		this->switchk = raw_pointer_cast(&switchk_shared[0]);
		this->global_kt = raw_pointer_cast(&global_kt_shared[0]);
		this->overflow = raw_pointer_cast(&overflow_shared[0]);
		this->isend = raw_pointer_cast(&isend_shared[0]);

	}

};

// A group of local queues of node IDs, used by an entire thread block.
// Multiple queues are used to reduce memory contention.
// Thread i uses queue number (i % NUM_BIN).
template <class Ktype>
struct LocalQueues {
	// tail[n] is the index of the first empty array in elems[n]
	int tail[NUM_BIN];

	// Queue elements.
	// The contents of queue n are elems[n][0 .. tail[n] - 1].
	Ktype elems[NUM_BIN][W_QUEUE_SIZE];

	// The number of threads sharing queue n.  We use this number to
	// compute a reduction over the queue.
	int sharers[NUM_BIN];

	// Initialize or reset the queue at index 'index'.
	// Normally run in parallel for all indices.
	__device__ void reset(int index, dim3 block_dim) {
		tail[index] = 0;		// Queue contains nothing

		// Number of sharers is (threads per block / number of queues)
		// If division is not exact, assign the leftover threads to the first
		// few queues.
		sharers[index] =
				(block_dim.x >> EXP) +   // block_dim/8
				(threadIdx.x < (block_dim.x & MOD_OP));
	}

	// Append 'value' to queue number 'index'.  If queue is full, the
	// append operation fails and *overflow is set to 1.
	__device__ void append(int index, int *overflow, Ktype value) {
		// Queue may be accessed concurrently, so
		// use an atomic operation to reserve a queue index.
		int tail_index = atomicAdd(&tail[index], 1);
		if (tail_index >= W_QUEUE_SIZE)
			*overflow = 1;
		else
			elems[index][tail_index] = value;
	}

	// Perform a scan on the number of elements in queues in a a LocalQueue.
	// This function should be executed by one thread in a thread block.
	//
	// The result of the scan is used to concatenate all queues; see
	// 'concatenate'.
	//
	// The array prefix_q will hold the scan result on output:
	// [0, tail[0], tail[0] + tail[1], ...]
	//
	// The total number of elements is returned.
	__device__ int size_prefix_sum(int (&prefix_q)[NUM_BIN]) {
		prefix_q[0] = 0;
		for(int i = 1; i < NUM_BIN; i++){
			prefix_q[i] = prefix_q[i-1] + tail[i-1];
		}
		return prefix_q[NUM_BIN-1] + tail[NUM_BIN-1];
	}

	// Concatenate and copy all queues to the destination.
	// This function should be executed by all threads in a thread block.
	//
	// prefix_q should contain the result of 'size_prefix_sum'.
	__device__ void concatenate(Ktype *dst, int (&prefix_q)[NUM_BIN]) {
		// Thread n processes elems[n % NUM_BIN][n / NUM_BIN, ...]
		int q_i = threadIdx.x & MOD_OP; // w-queue index, idx of row
		int local_shift = threadIdx.x >> EXP; // shift within a w-queue, idx of col

		while(local_shift < tail[q_i]){
			dst[prefix_q[q_i] + local_shift] = elems[q_i][local_shift];

			//multiple threads are copying elements at the same time,
			//so we shift by multiple elements for next iteration
			local_shift += sharers[q_i]; // 8*64>512, so it is out of bound???
		}
	}
};



// Process a single graph node from the active frontier.  Mark nodes and
// put new frontier nodes on the queue.
//
// 'pid' is the ID of the node to process.
// 'index' is the local queue to use, chosen based on the thread ID.
// The output goes in 'local_q' and 'overflow'.
// Other parameters are inputs.
__device__ void
visit_node(int pid,
		int index,
		LocalQueues<int> &local_q,
		int *overflow,
		Node *g_graph_nodes,
		Edge *g_graph_edges,
		int *g_color,
		int *g_cost,
		int gray_shade,
		int endId,
		bool *is_end,
		int *d_cameFrom)
{
	g_color[pid] = BLACK;		// Mark this node as visited
	int cur_cost = g_cost[pid];	// Look up shortest-path distance to this node
//	Node cur_node = tex1Dfetch(g_graph_node_ref,pid);
	Node cur_node = g_graph_nodes[pid];
//	Node cur_node = ugraph.graph_nodeD[pid];
	// if terminal node is the end
	//  if(pid==endId)
	//  {
	//  	*is_end=true;
	//  	return;
	//  }
	// For each outgoing edge
	for(int i = cur_node.x; i < cur_node.y + cur_node.x; i++) {
		Edge cur_edge = g_graph_edges[i];
		int id = cur_edge.x;  // terminal node id---nbr
		int cost = cur_edge.y;
		cost += cur_cost;
		int orig_cost = atomicMin(&g_cost[id],cost);

		// If this outgoing edge makes a shorter path than any previously
		// discovered path
		if(orig_cost > cost){
			int old_color = atomicExch(&g_color[id],gray_shade);
			if(old_color != gray_shade) {   // not in this level
				//push to the queue
				local_q.append(index, overflow, id);  // push id to queue[index]
				d_cameFrom[id]=pid;
			}
		}
	}
}

//-------------------------------------------------
//This is the version for one-block situation. The propagation idea is basically the same as
//BFS_kernel.
//The major differences are:
// 1) This kernel can propagate though multiple BFS levels (while loop) using __synchThreads() between levels
// 2) the intermediate queues are stored in shared memory (next_wf)
//\param q1: the current frontier queue when the kernel is launched
//\param q2: the new frontier queue when the  kernel returns
//--------------------------------------------------
template <class Ktype>
__global__ void
BFS_in_GPU_kernel(ubfsGraphBase<int> ugraph,
		Ktype *q1,
		Ktype *q2,
		Node *g_graph_nodes,
		Edge *g_graph_edges,
		int no_of_nodes,
		int gray_shade,
		int k,
		int endId,
		int* rounds,
		int* d_cameFrom)
{
	__shared__ LocalQueues<Ktype> local_q;
	__shared__ int prefix_q[NUM_BIN];// store number of elems of each rows

	//next/new wave front
	__shared__ Ktype next_wf[MAX_THREADS_PER_BLOCK];
	__shared__ int  tot_sum;
	if(threadIdx.x == 0)
		tot_sum = 0;//total number of new frontier nodes
	while(1){//propage through multiple BFS levels until the wavfront overgrows one-block limit
		if(threadIdx.x < NUM_BIN){
			local_q.reset(threadIdx.x, blockDim);
		}
		__syncthreads();
		int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
		if( tid<no_of_nodes)
		{
			Ktype pid;  // id of nodes in frontier
			if(tot_sum == 0)//this is the first BFS level of current kernel call
				pid = q1[tid];
			else
				pid = next_wf[tid];//read the current frontier info from last level's propagation

			// Visit a node from the current frontier; update costs, colors, and
			// output queue (only int)
			visit_node(pid, threadIdx.x & MOD_OP, local_q, ugraph.overflow,g_graph_nodes,g_graph_edges,
					ugraph.colorD, ugraph.costD, gray_shade,endId,ugraph.isend,d_cameFrom);

		}
		__syncthreads();
		if(threadIdx.x == 0){
			*(ugraph.tail) = tot_sum = local_q.size_prefix_sum(prefix_q);
		}
		__syncthreads();

		if(tot_sum == 0)//the new frontier becomes empty; BFS is over
			return;
		if(tot_sum <= MAX_THREADS_PER_BLOCK){
			//the new frontier is still within one-block limit;
			//stay in current kernel
			local_q.concatenate(next_wf, prefix_q);
			__syncthreads();
			no_of_nodes = tot_sum;
			if(threadIdx.x == 0){
				if(gray_shade == GRAY0)
					gray_shade = GRAY1;
				else
					gray_shade = GRAY0;
			}
		}
		else{
			//the new frontier outgrows one-block limit; terminate current kernel
			local_q.concatenate(q2, prefix_q);
			return;
		}
		if(*(ugraph.isend))
		{
			return;
		}
		if(threadIdx.x == 0)
			rounds[0]++;
	}//while

}

volatile __device__ int count = 0;
volatile __device__ int no_of_nodes_vol = 0;
volatile __device__ int stay_vol = 0;

//Inter-block sychronization
//This only works when there is only one block per SM
__device__ void start_global_barrier(int fold){
	__syncthreads();

	if(threadIdx.x == 0){
		atomicAdd((int*)&count, 1);
		while( count < NUM_SM*fold){
			;
		}
	}
	__syncthreads();

}

__device__ void rst_barrier(int *global_kt)
{
	count = 0;
	*global_kt=0;
}
/*****************************************************************************
  This BFS kernel propagates through multiple levels using global synchronization
  The basic propagation idea is the same as "BFS_kernel"
  The major differences are:
  1) propagate through multiple levels by using GPU global sync ("start_global_barrier")
  2) use q1 and q2 alternately for the intermediate queues
  \param q1: the current frontier when the kernel is called
  \param q2: possibly the new frontier when the kernel returns depending on how many levels of propagation
            has been done in current kernel; the new frontier could also be stored in q1
  \param switch_k: whether or not to adjust the "k" value on the host side
                Normally on the host side, when "k" is even, q1 is the current frontier; when "k" is
                odd, q2 is the current frontier; since this kernel can propagate through multiple levels,
                the k value may need to be adjusted when this kernel returns.
  \param global_kt: the total number of global synchronizations,
                    or the number of times to call "start_global_barrier"
 *****************************************************************************/
template <class Ktype>
__global__ void
BFS_kernel_multi_blk_inGPU(ubfsGraphBase<int> ugraph,
		Ktype *q1,
		Ktype *q2,
		Node *g_graph_nodes,
		Edge *g_graph_edges,
		int *no_of_nodes,   //the  frontier num
		int gray_shade,
		int k,
		int endId,
		int* d_cameFrom)
{
	__shared__ LocalQueues<Ktype> local_q;
	__shared__ int prefix_q[NUM_BIN];
	__shared__ int shift;
	__shared__ int no_of_nodes_sm;
	__shared__ int odd_time;// the odd level of propagation within current kernel
	if(threadIdx.x == 0){
		odd_time = 1;//true;
		if(blockIdx.x == 0)
			no_of_nodes_vol = *no_of_nodes;
	}
	int kt = atomicOr(ugraph.global_kt,0);// the total count of GPU global synchronization
	while (1){//propagate through multiple levels
		if(threadIdx.x < NUM_BIN){
			local_q.reset(threadIdx.x, blockDim);
		}
		if(threadIdx.x == 0)
			no_of_nodes_sm = no_of_nodes_vol;
		__syncthreads();

		int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
		if( tid<no_of_nodes_sm)
		{
			// Read a node ID from the current input queue
			Ktype *input_queue = odd_time ? q1 : q2;
			Ktype pid= input_queue[tid];
//			int pid = atomicOr((int *)&input_queue[tid], 0);  //????????????????

			// Visit a node from the current frontier; update costs, colors, and
			// output queue
			visit_node(pid, threadIdx.x & MOD_OP, local_q, ugraph.overflow,g_graph_nodes,g_graph_edges,
					ugraph.colorD, ugraph.costD, gray_shade,endId,ugraph.isend,d_cameFrom);
			if(*ugraph.isend)
			{
				rst_barrier(ugraph.global_kt);
				return;
			}
		}
		__syncthreads();

		// Compute size of the output and allocate space in the global queue
		if(threadIdx.x == 0){
			int tot_sum = local_q.size_prefix_sum(prefix_q);
			shift = atomicAdd(ugraph.tail, tot_sum);
		}
		__syncthreads();

		// Copy to the current output queue in global memory
		Ktype *output_queue = odd_time ? q2 : q1;
		local_q.concatenate(output_queue + shift, prefix_q);

		if(threadIdx.x == 0){
			odd_time = (odd_time+1)%2;
			if(gray_shade == GRAY0)
				gray_shade = GRAY1;
			else
				gray_shade = GRAY0;
		}

		//synchronize among all the blks
		start_global_barrier(kt+1);
		if(blockIdx.x == 0 && threadIdx.x == 0){
			stay_vol = 0;
			if(*(ugraph.tail)< NUM_SM*MAX_THREADS_PER_BLOCK && *(ugraph.tail) > MAX_THREADS_PER_BLOCK){
				stay_vol = 1;
				no_of_nodes_vol = *(ugraph.tail);
				*(ugraph.tail) = 0;
			}
		}
		start_global_barrier(kt+2);
		kt+= 2;
		if(stay_vol == 0)
		{
			if(blockIdx.x == 0 && threadIdx.x == 0)
			{
				rst_barrier(ugraph.global_kt);
//				*global_kt = kt;
//				*(ugraph.global_kt)= kt;
				*(ugraph.switchk) = (odd_time+1)%2;
				*no_of_nodes = no_of_nodes_vol;
			}
			return;
		}
	}
}

/*****************************************************************************
  This is the  most general version of BFS kernel, i.e. no assumption about #block in the grid
  \param q1: the array to hold the current frontier
  \param q2: the array to hold the new frontier
  \param g_graph_nodes: the nodes in the input graph
  \param g_graph_edges: the edges i nthe input graph
  \param g_color: the colors of nodes
  \param g_cost: the costs of nodes
  \param no_of_nodes: the number of nodes in the current frontier
  \param tail: pointer to the location of the tail of the new frontier. *tail is the size of the new frontier
  \param gray_shade: the shade of the gray in current BFS propagation. See GRAY0, GRAY1 macro definitions for more details
  \param k: the level of current propagation in the BFS tree. k= 0 for the first propagation.
 ***********************************************************************/
template <class Ktype>
__global__ void
BFS_kernel(ubfsGraphBase<int> ugraph,
		Ktype *q1,
		Ktype *q2,
		Node *g_graph_nodes,
		Edge *g_graph_edges,
		int no_of_nodes,
		int gray_shade,
		int k,
		int endId,
		int* d_cameFrom)
{
	__shared__ LocalQueues<Ktype> local_q;
	__shared__ int prefix_q[NUM_BIN];//the number of elementss in the w-queues ahead of
	//current w-queue, a.k.a prefix sum
	__shared__ int shift;

	if(threadIdx.x < NUM_BIN){
		local_q.reset(threadIdx.x, blockDim);
	}
	__syncthreads();

	//first, propagate and add the new frontier elements into w-queues
	int tid = blockIdx.x*MAX_THREADS_PER_BLOCK + threadIdx.x;
	if( tid < no_of_nodes)
	{
		// Visit a node from the current frontier; update costs, colors, and
		// output queue
		visit_node(q1[tid], threadIdx.x & MOD_OP, local_q, ugraph.overflow,g_graph_nodes,g_graph_edges,
				ugraph.colorD, ugraph.costD, gray_shade,endId,ugraph.isend,d_cameFrom);
		if(*ugraph.isend)
		{
			return;
		}
	}
	__syncthreads();

	// Compute size of the output and allocate space in the global queue
	if(threadIdx.x == 0){
		//now calculate the prefix sum
		int tot_sum = local_q.size_prefix_sum(prefix_q);
		//the offset or "shift" of the block-level queue within the
		//grid-level queue is determined by atomic operation
		shift = atomicAdd(ugraph.tail,tot_sum);
//		shift = atomicAdd(tail,tot_sum);
	}
	__syncthreads();

	//now copy the elements from w-queues into grid-level queues.
	//Note that we have bypassed the copy to/from block-level queues for efficiency reason
	local_q.concatenate(q2 + shift, prefix_q);
}





}
