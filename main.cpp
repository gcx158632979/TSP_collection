#include <iostream>
#include <string>
#include <vector>
#include <sstream> //字符串转换
#include <fstream>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <bits/stdc++.h>
#include <numeric>
#include <algorithm>
using namespace std;
class GA
{
private:
    int num_city_;
    int num_total_;
    int iteration_;
    vector<vector<double>> dis_mat_;
    vector<vector<int>> fruits_;
    vector<double> scores_;
    vector<int> init_best_;
    vector<int> iter_x_;
    vector<double> iter_y_;
    double ga_choose_ratio_;
    double mutate_ratio_;
public:
    GA(int num_city, int num_total, int iteration, vector<vector<double>> waypoints);
    ~GA();
    vector<vector<double>> compute_dis_mat(int num_city, vector<vector<double>> waypoints);
    vector<vector<int>> greedy_init();
    vector<double> compute_adp(vector<vector<int>> fruits);
    double compute_pathlen(vector<int> fruit);
    std::pair<vector<int>,double> run();
    std::pair<vector<int>, double> ga();
    std::pair<vector<vector<int>>, vector<double>> ga_parent(vector<double> scores);
    std::pair<vector<int>, vector<int>> ga_choose(vector<double> parents_score, vector<vector<int>> parents);
    std::pair<vector<int>, vector<int>> ga_cross(vector<int> gene1, vector<int> gene2);
    vector<int> ga_mutate(vector<int> gene_origin);
};

GA::GA(int num_city, int num_total, int iteration, vector<vector<double>> waypoints)
{
    
    num_city_ = num_city;
    num_total_ = num_total;
    iteration_ = iteration;
    ga_choose_ratio_ = 0.2;
    mutate_ratio_ = 0.05;
    dis_mat_ = compute_dis_mat(num_city, waypoints);
    fruits_ = greedy_init();
    scores_ = compute_adp(fruits_);
    int maxPosition = max_element(scores_.begin(),scores_.end()) - scores_.begin(); 

    init_best_ = fruits_[maxPosition];
    
    iter_x_.emplace_back(0);
    iter_y_.emplace_back(1.0/scores_[maxPosition]);
}
vector<int> GA::ga_mutate(vector<int> gene_origin)
{
    vector<int> gene_new(gene_origin);
    int start = rand()%num_total_;
    int end = -1;
    while (end < start)
    {
        end = rand()%num_total_;
    }
    reverse(gene_new.begin()+start,gene_new.begin()+end);
    return gene_new;
}
std::pair<vector<int>, vector<int>> GA::ga_cross(vector<int> gene1, vector<int> gene2)
{
    int start = rand()%num_total_;
    int end = -1;
    while (end < start)
    {
        end = rand()%num_total_;
    }
    vector<int> x_conflict_index;
    for (int i =start;i<end;i++){
        int tmp = gene1[i];
        vector<int>::iterator iter=std::find(gene2.begin(),gene2.end(),tmp);
        int index = distance(gene2.begin(),iter);
        if (!(index>=start&&index<end)){
            x_conflict_index.emplace_back(index);
        }
    }
    vector<int> y_conflict_index;
    for (int i=start;i<end;i++){
        int tmp = gene2[i];
        vector<int>::iterator iter=std::find(gene1.begin(),gene1.end(),tmp);
        int index = distance(gene1.begin(),iter);
        if (!(index>=start&&index<end)){
            y_conflict_index.emplace_back(index);
        }
    }
    vector<int> tmp;
    tmp.assign(gene1.begin()+start,gene1.begin()+end);
    gene1.erase(gene1.begin()+start,gene1.begin()+end);
    gene1.insert(gene1.begin()+start,gene2.begin()+start,gene2.begin()+end);
    gene2.erase(gene2.begin()+start,gene2.begin()+end);
    gene2.insert(gene2.begin()+start,tmp.begin(),tmp.end());
    for (int i=0;i<x_conflict_index.size();i++){
        int x_index = x_conflict_index[i];
        int y_index = y_conflict_index[i];
        int tmp = gene2[x_index];
        gene2[x_index] = gene1[y_index];
        gene1[y_index] = tmp;
    }
    return std::make_pair(gene1,gene2);
}
std::pair<vector<int>, vector<int>> GA::ga_choose(vector<double> parents_score, vector<vector<int>> parents)
{
    double sum = accumulate(parents_score.begin(),parents_score.end(),0);
    double rand1 = rand() % (1000) / 1000.0;
    double rand2 = rand() % (1000) / 1000.0;
    int index1,index2;
    for (int i =0;i<parents_score.size();i++){
        double score_ratio = parents_score[i] / sum;
        if (rand1>=0){
            rand1 = rand1 - score_ratio;
            if (rand1<=0){
                index1 = i;
            }
        }
        if (rand2>=0){
            rand2 = rand2 - score_ratio;
            if (rand2<=0){
                index2 = i;
            }
        }
        if (rand1 < 0 && rand2 < 0){
            break;
        }
    }
    return std::make_pair(parents[index1],parents[index2]);
}
std::pair<vector<vector<int>>, vector<double>> GA::ga_parent(vector<double> scores)
{
    vector<vector<int>> parents;
    vector<double> parents_score;
    vector<vector<int>> fruits_copy(fruits_);
    vector<double> scores_copy(scores);
    //cout<<num_total<<endl;
    for (int i = 0;i< int(ga_choose_ratio_*num_total_);i++){
        int maxPosition = max_element(scores_copy.begin(),scores_copy.end()) - scores_copy.begin(); 
        parents.emplace_back(fruits_copy[maxPosition]);
        parents_score.emplace_back(scores_copy[maxPosition]);
        fruits_copy.erase(fruits_copy.begin()+maxPosition);
        scores_copy.erase(scores_copy.begin()+maxPosition);
    }
    return std::make_pair(parents, parents_score);
}
std::pair<vector<int>, double> GA::ga()
{
    vector<double> scores = compute_adp(fruits_);
    std::pair<vector<vector<int>>, vector<double>> result = ga_parent(scores);
    vector<vector<int>> parents = result.first;
    vector<double> parents_score =result.second;
    vector<int> tmp_best_one = parents[0];
    double tmp_best_score = parents_score[0];
    vector<vector<int>> fruits_new(parents);
    while (fruits_new.size()<num_total_)
    {
        std::pair<vector<int>, vector<int>> gene_pair = ga_choose(parents_score, parents);
        std::pair<vector<int>, vector<int>> gene_pair_new = ga_cross(gene_pair.first, gene_pair.second);
        vector<int> gene_x_new, gene_y_new;
        if (rand() % (1000) / 1000.0 < mutate_ratio_){
            gene_x_new = ga_mutate(gene_pair_new.first);
        }
        else{
            gene_x_new = gene_pair_new.first;
        }
        if (rand() % (1000) / 1000.0 < mutate_ratio_){
            gene_y_new = ga_mutate(gene_pair_new.second);
        }
        else{
            gene_y_new = gene_pair_new.second;
        }
        double x_adp = 1.0 / compute_pathlen(gene_x_new);
        double y_adp = 1.0 / compute_pathlen(gene_y_new);
        if (x_adp > y_adp){
            if (std::find(fruits_new.begin(), fruits_new.end(), gene_x_new) == fruits_new.end()){
                fruits_new.emplace_back(gene_x_new);
            }
        }
        else if (y_adp >= x_adp)
        {
            if (std::find(fruits_new.begin(), fruits_new.end(), gene_y_new) == fruits_new.end()){
                fruits_new.emplace_back(gene_y_new);
            }
        }
    }
    fruits_.clear();
    fruits_.assign(fruits_new.begin(),fruits_new.end());
    return std::make_pair(tmp_best_one, tmp_best_score);
}
std::pair<vector<int>,double> GA::run()
{
    vector<int> BEST_LIST;
    double best_score = -DBL_MAX;
    for (int i =0;i<iteration_+1;i++){
        std::pair<vector<int>, double> result = ga();
        vector<int> tmp_best_one = result.first;
        double tmp_best_score = result.second;
        iter_x_.emplace_back(i);
        iter_y_.emplace_back(1.0/tmp_best_score);
        if (tmp_best_score > best_score){
            best_score = tmp_best_score;
            BEST_LIST.assign(tmp_best_one.begin(), tmp_best_one.end());
        }
        cout<<i<<" "<<1./best_score<<endl;
    }
    return std::make_pair(BEST_LIST, 1.0/best_score);
}
vector<double> GA::compute_adp(vector<vector<int>> fruits)
{
    vector<double> adp;
    for (int i = 0;i < fruits.size();i++){
        double length = compute_pathlen(fruits[i]);
        adp.emplace_back(1.0/length);
    }
    return adp;
}
double GA::compute_pathlen(vector<int> fruit)
{
    double result = 0;
    for (int i =0;i<fruit.size()-1;i++){
        int a = fruit[i];
        int b = fruit[i+1];
        result = result + dis_mat_[a][b];
    }
    return result;
}
vector<vector<int>> GA::greedy_init()
{
    int start_idx = 0;
    vector<vector<int>> result;
    for (int i = 0; i < num_total_; i++){
        vector<int> rest;
        for (int j = 0;j < num_city_;j++){
            rest.emplace_back(j);
        }
        if (start_idx >= num_city_){
            start_idx = rand() % num_city_;
            vector<int> tmp(result[start_idx]);
            result.emplace_back(tmp);
            continue;
        }
        int current = start_idx;
        rest.erase(rest.begin()+current);
        vector<int> result_one;
        result_one.emplace_back(current);
        while (rest.size()!=0)
        {
            double tmp_min = DBL_MAX;
            double tmp_choose = -1;
            for (int x=0;x<rest.size();x++){
                if (dis_mat_[current][rest[x]] < tmp_min){
                    tmp_min = dis_mat_[current][rest[x]];
                    tmp_choose = rest[x];
                }
            }
            current = tmp_choose;
            result_one.emplace_back(tmp_choose);
            vector<int>::iterator iter=find(rest.begin(),rest.end(),tmp_choose);
            rest.erase(iter);
        }
        result.emplace_back(result_one);
        start_idx = start_idx + 1;
    }
    return result;
}
vector<vector<double>> GA::compute_dis_mat(int num_city, vector<vector<double>> waypoints)
{
    double dis_mat[num_city][num_city];
    for (int i = 0; i < num_city; i++){
        for (int j = 0; j < num_city; j++){
            if (i==j){
                dis_mat[i][j] = DBL_MAX;
                continue;
            }
            double city1_x = waypoints[i][0];
            double city1_y = waypoints[i][1];
            double city2_x = waypoints[j][0];
            double city2_y = waypoints[j][1];
            double distance = sqrt(pow(city1_x-city2_x,2)+pow(city1_y-city2_y,2));
            dis_mat[i][j] = distance;
        }
    }
    vector<vector<double>> result;
    for (int i = 0; i < num_city; i++){
        vector<double> tmp;
        for (int j = 0; j < num_city; j++){
            tmp.emplace_back(dis_mat[i][j]);
        }
        result.emplace_back(tmp);
    }
    return result;
}
GA::~GA()
{
}

int main()
{
    ifstream fin("C:/Users/gcx/Desktop/TSP_collection/data/st70.txt");
	string line;
	vector<vector<double>> waypoints;
    
	if (fin)
	{
		while (getline(fin, line)) //按行读取到line_info中 
    	{
			if (line.size() == 0)
			{
				break;   //循环体内手动的进行空行的判断
			}
			//cout << "line:" << line << endl;
			istringstream sin(line); //create string input object
			vector<string> Waypoints;
			string info;
 
			while (getline(sin, info, ' '))
			{
				//cout << "info:" << info << endl;
				Waypoints.push_back(info);
			}
 
			string x2_str = Waypoints[1];
			string x3_str = Waypoints[2];
			// cout<< "x_str" << x_str << endl;
			// cout<< "y_str" << y_str << endl;
 
			double x2, x3;
			stringstream sx1, sx2, sx3; //transform string to double
 
			sx2<< x2_str;
			sx3 << x3_str;
			sx2 >> x2;
			sx3>> x3;
            vector<double> location_tmp;
			location_tmp.push_back(x2);
            location_tmp.push_back(x3);
            waypoints.emplace_back(location_tmp);
		}
	}
	else
	{
		cout << "no such file" << endl;;
	}
	fin.close();
    GA test(waypoints.size(),25,50,waypoints);
    std::pair<vector<int>,double> result=test.run();
    // for (int i =0;i<result.first.size();i++){
    //     for (int j=0;j<waypoints[result.first[i]].size();j++){
    //         cout<<waypoints[result.first[i]][j]<<" ";
    //     }
    //     cout<<endl;
    // }
	return 0;
}