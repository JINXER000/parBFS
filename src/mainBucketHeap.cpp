#include <iostream>
#include <assert.h>
#include <list>
#include <chrono>
#include <fstream>


#include "utils.h"
#include "bucketedqueue.h"
# include "ubfsfunc.h"


using namespace std;
using namespace std::chrono;



//#define USE_GPU


int main(){

	ifstream inputFile;
	ofstream outputFile;
	string inputFileName;
	bool nonDirectedGraph = false;

	int startVertex = 65, destination = 1;
	int numVertices,numEdges;
		int total_rounds=0;

	inputFileName = "input/NetworkScienceMinified.txt";
	openFileToAccess< std::ifstream >( inputFile, inputFileName );
	if( !inputFile.is_open()) {
		std::cerr << "input file not found " << std::endl;
		throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
	}

	//version2
	//    std::vector<initial_vertex> parsedGraph( 0 );
	//    numEdges = parse_graph(
	//            inputFile,		// Input file.
	//            parsedGraph,	// The parsed graph.
	//            startVertex,
	//            nonDirectedGraph );		// Arbitrary user-provided parameter.
	//
	//    numVertices= parsedGraph.size();




	// version 1
	//     if the first line of input file specifies num of vertices and edges
	std::string line;
	char delim[3] = " \t";	//In most benchmarks, the delimiter is usually the space character or the tab character.
	char* pch;
	std::getline( inputFile, line );
	char cstrLine[256];
	std::strcpy( cstrLine, line.c_str() );

	pch = strtok(cstrLine, delim);
	numVertices = atoi( pch );
	pch = strtok( NULL, delim );
	numEdges = atoi( pch );

	list<struct AdjacentNode>* adjList = new list<struct AdjacentNode>[numVertices];

	int s,v,w;
	for (int i = 0 ; i < numEdges ; i++){
		// cin >> s >> v >> w;

		if(!std::getline( inputFile, line ))
			break;
		std::strcpy( cstrLine, line.c_str() );

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			s = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			v = atoi( pch );
		else
			continue;
		pch=strtok( NULL, delim );
		if( pch != NULL )
			w = atoi( pch );
		adjList[s].push_back({.terminalVertex=v, .weight=w});
	}


	//  int* distance = new int[numVertices];
	std::vector<int> distance(numVertices);
	std::vector<int> cameFrom(numVertices);
#ifndef USE_GPU

	auto start = high_resolution_clock::now();

	// another implementation
	BucketPrioQueue<int> open;
	for (int i = 0 ; i < numVertices ; i++) {
		if (i == startVertex){
			open.push(0,i);
			distance[i] = 0;
			cameFrom[i] = i;
		} else {
			open.push(INT_MAX-1,i);
			distance[i] = INT_MAX-1;
			cameFrom[i] = INT_MAX-1;
		}
//		total_rounds++;
	}
	while (!open.empty()) {
		int cur_key = open.pop();
		total_rounds++;
		if (cur_key == destination) break;
		for (struct AdjacentNode n : adjList[cur_key]) {
			if (distance[n.terminalVertex] > distance[cur_key] + n.weight) {
				distance[n.terminalVertex] = distance[cur_key] + n.weight;
				open.push(distance[n.terminalVertex],n.terminalVertex);
				cameFrom[n.terminalVertex] = cur_key;
			}
		}
	}
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "duration is "<<duration.count() << endl;
	cout << "distance is "<<distance[destination] << endl;
	std::cout<<"total rounds= "<<total_rounds<<std::endl;
	// retrace
	std::vector<int> search_path;
	int ptOnPath=destination;
	while(cameFrom[ptOnPath]!= ptOnPath)
	{
		search_path.push_back(ptOnPath);
		ptOnPath= cameFrom[ptOnPath];
	}
//	std::reverse(search_path.begin(),search_path.end());
	for(auto ptr:search_path)
	{
		std::cout<<ptr<<",";
	}
#else


	Graph<AdjacentNode> cuGraph;
	cuGraph.numEdges=numEdges;
	cuGraph.numVertices=numVertices;
	for (int i = 0; i < numVertices; i++) {
		cuGraph.edgesOffset.push_back(cuGraph.adjacencyList.size());
		cuGraph.edgesSize.push_back(adjList[i].size());
		for (auto &edge: adjList[i]) {
			cuGraph.adjacencyList.push_back(edge);
		}
	}
	// GPU test
	std::vector<int> srcNode;
	for (int i = 0 ; i < numVertices ; i++) {
		if (i == startVertex){
			srcNode.push_back(i);
			distance[i] = 0;
		}else
		{
			distance[i] = INT_MAX-1;
		}
	}
//		int inputSize=78;
//		for(int i=0;i<inputSize;i++)
//			{
//			if(i>5&&i<10)
//				srcNode.push_back(0);
//			else
//				srcNode.push_back(i+1);
//		}
	auto start = high_resolution_clock::now();
	ubfs::parWavefront(srcNode,cuGraph,distance,cameFrom,destination);

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	std::cout <<"duration is "<< duration.count() << std::endl;

#endif
	return 0;
}
