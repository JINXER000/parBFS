/*
 * ubfsfunc.h
 *
 *  Created on: Apr 24, 2020
 *      Author: yzchen
 */

#ifndef UBFSFUNC_H_
#define UBFSFUNC_H_

#include "utils.h"
namespace ubfs{
void parWavefront(std::vector<int> &srcNode,
		Graph<AdjacentNode> &cuGraph,
		std::vector<int> &distances,
		std::vector<int> &cameFrom,
		int destination);
}




#endif /* UBFSFUNC_H_ */
