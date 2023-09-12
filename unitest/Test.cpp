//
// Created by kingkiller on 2023/9/9.
//
#include <iostream>
#include <vector>

int main()
{
    int outs = 5;
    std::vector<std::vector<float>> datas(outs);
    for (int i = 0; i < outs; ++i) {
        datas[i].assign(5, 0.0f);
    }

    for (auto vec : datas) {
        for (auto item : vec) {
            std::cout << item << std::endl;
        }
    }
}