//
// Created by KYL.ai on 2021-03-17.
//

#ifndef FACETOOL_UTILS_H
#define FACETOOL_UTILS_H
#include <fstream>

/**
 * read label file and return labels as vector of string
 * @param file_name : label file path
 * @param labels : vector of string labels
 */
void read_weights(char *filename, cv::Mat dst)
{
    std::ifstream readFile;
    readFile.open(filename);
    if (readFile.is_open())
    {
        int i = 0;
        while (!readFile.eof())
        {
            std::string str;
            std::getline(readFile, str);
            if (i < 204)
            {
                std::istringstream iss(str);
                std::string temp;
                int j = 0;
                while (getline(iss, temp, ','))
                {
                    dst.at<float>(i, j) = std::stof(temp);
                    j++;
                }
            }
            i++;
        }
        readFile.close();
    }
}

#endif //FACETOOL_UTILS_H
