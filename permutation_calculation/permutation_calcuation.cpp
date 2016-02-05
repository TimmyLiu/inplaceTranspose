#include <vector>
#include <iostream>
using namespace std;
typedef unsigned long long ullong;

void permutation_calculation(ullong m, ullong n, vector<vector<ullong>> &permutationVec)
{
	/*
	calculate inplace transpose permutation lists
	reference:
	https://en.wikipedia.org/wiki/In-place_matrix_transposition
	and 
	http://www.netlib.org/utk/people/JackDongarra/CCDSC-2014/talk35.pdf
	row major matrix of size n x m
	p(k) = (k*n)mod(m*n-1), if 0 < k < m*n-1
	when k = 0 or m*n-1, it does not require movement
	*/
	if (m < 1 || n < 1)
		return;
	ullong mn_minus_one = m*n - 1;
	//maintain a table so check is faster
	ullong *table = new ullong[mn_minus_one+1]();//init to zeros
	table[0] += 1;
	//i should probably use map of maps
	for (ullong i = 1; i < mn_minus_one; i++)
	{
		//first check if i is already stored in somewhere in vector of vectors
		bool already_checked = false;

		/*
		//this check is too slow; maintain a table instead
		for (int vector_idx = 0; vector_idx < permutationVec.size(); vector_idx++)
		{
			vector<int>::iterator itor = find(permutationVec[vector_idx].begin(), permutationVec[vector_idx].end(), i);
			if (itor != permutationVec[vector_idx].end())
			{
				already_checked = true;
				break;
			}
		}
		*/

		if (table[i] >= 1)
			already_checked = true;
		if (already_checked == true)
			continue;

		//if not checked yet
		vector<ullong> vec;
		vec.push_back(i);
		table[i] += 1;
		vector<ullong>::iterator itor;
		ullong temp = i;
		while (true)
		{
			temp = (temp*n);
			temp = temp % (mn_minus_one);
			if (find(vec.begin(), vec.end(), temp) != vec.end())
			{
				//what goes around comes around JT
				break;
			}
			if (temp > mn_minus_one)
			{
				cout << "weird. out of bound." << endl;
			}
			if (table[temp] >= 1)
			{
				already_checked = true;
				break;
			}
			vec.push_back(temp);
			table[temp] += 1;
		}
		if (already_checked == true)
			continue;
		permutationVec.push_back(vec);
	}
	delete[] table;
}

int main()
{
	vector<vector<ullong>> permutationVec;

	permutation_calculation(2048, 1024, permutationVec);

	//print
	int size = 0;
	std::cout << "permutation has " << permutationVec.size() << " vectors" << std::endl;
	for (int i = 0; i < permutationVec.size(); i++)
	{
		size = size + permutationVec[i].size();
		//std::cout << "permutation "<< i << " has " << permutationVec[i].size() << " elements" << std::endl;
	}
	cout << "overall size is " << size << endl;

	//print the first element of each vector
	for (int i = 0; i < permutationVec.size(); i++)
	{
		std::cout << "{" << permutationVec[i][0] << "}," << std::endl;
	}
}