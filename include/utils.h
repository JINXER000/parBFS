#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <vector>
#include <cstdlib>
#include <sys/time.h>
#include <string>
#include <cstring>
#include <stdio.h>
#include <iostream>
using namespace std;

template <typename Vtype>
struct Graph {
    std::vector<Vtype> adjacencyList; // all edges
    std::vector<int> edgesOffset; // offset to adjacencyList for every vertex
    std::vector<int> edgesSize; //number of edges for every vertex
    int numVertices = 0;
    int numEdges = 0;
};
struct edge{
        int src;
        int dest;
        int w;
};

class neighbor {
public:
        unsigned int weight;
        unsigned int dstIndex;
};

class initial_vertex {
public:
	unsigned int distance;
	std::vector<neighbor> nbrs;
	initial_vertex():nbrs(0){}
};

class Timer {
	struct timeval startingTime;
public:
	void set();
	double get();
};

uint parse_graph(
	std::ifstream& inFile,
	std::vector<initial_vertex>& initGraph,
	const long long arbparam,
	const bool nondirected );

void sort_by_dest(edge* edges, int nEdges, vector<initial_vertex> * peeps);
void sort_by_src(edge* edges, int nEdges);
void testCorrectness(edge* edges, int* results, int nEdges, int nNodes);
void saveResults(ofstream& outFile, int* solution, int n);

struct AdjacentNode {
    int terminalVertex;
    int weight;
};
// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}
#endif	//	PARSE_GRAPH_HPP
