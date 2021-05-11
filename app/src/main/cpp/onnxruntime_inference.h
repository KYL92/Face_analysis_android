//
// Created by KYL.ai on 2021-03-17.
//

#ifndef FACETOOL_ONNXRUNTIME_INFERENCE_H
#define FACETOOL_ONNXRUNTIME_INFERENCE_H

#include <stdio.h>
#include <chrono>
#include <cinttypes>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include <opencv2/opencv.hpp>

using namespace cv;

#pragma comment(lib, "onnxruntime.lib")

class Inference {

private:

    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    unsigned long input_tensor_size;
    const char* modelpath_;
    const char* labelfilepath_;
    int img_height_;
    int img_width_;
    int origW_;
    int origH_;

    Mat u_base;
    Mat w_shape;
    Mat w_exp;
    Mat out_offset;
    Mat out_rotate;
    Mat out_shape;
    Mat out_expr;

    Ort::Value input_tensor{nullptr};

    std::vector<int64_t> input_node_dims;
    std::vector<int64_t> output_node_dims;
    std::vector<const char*> input_node_names;
    std::vector<const char*> output_node_names;
    std::unique_ptr<float[]> input_data_chw;
    std::unique_ptr<float[]> normalized;
    std::string predicted_speed;
    std::vector<std::array<float, 3>> output_vers;
    std::vector<float> symmetry;

    void createInputBuffer();
    void printNodes();
    void readForTDDFA(const char* model_path);

public:

    Inference(std::unique_ptr<Ort::Env>& env, const char*  modelpath,
              int img_height, int img_width, int ori_height, int ori_width, bool tddfa);
    Inference(const Inference& ) = delete;//no copy
    Inference& operator = (const Inference &) = delete;//no copy
    std::vector<std::array<float, 4>> detection(uint8_t* pixels);
    std::vector<std::array<float, 3>> TDDFA(uint8_t* pixels,int img_height, int img_width, int sx, int sy);
    Point gaze(Mat pixels, int width, int height);

    std::string getPredictedlabels();
    ~Inference();
};

// 3DDFA parameters
constexpr float param_std[62] = {0.000176, 0.000067, 0.000447, 26.550232, 0.000123, 0.000045,
                                 0.000079, 6.982563, 0.000435, 0.000123, 0.000174, 20.803040,
                                 575421.125000, 277649.062500, 258336.843750, 255163.125000,
                                 150994.375000, 160086.109375, 111277.304688, 97311.781250,
                                 117198.453125, 89317.367188, 88493.554688, 72229.929688,
                                 71080.210938, 50013.953125, 55968.582031, 47525.503906,
                                 49515.066406, 38161.480469, 44872.058594, 46273.238281,
                                 38116.769531, 28191.162109, 32191.437500, 36006.171875,
                                 32559.892578, 25551.117188, 24267.509766, 27521.398438,
                                 23166.531250, 21101.576172, 19412.324219, 19452.203125,
                                 17454.984375, 22537.623047, 16174.281250, 14671.640625,
                                 15115.688477, 13870.073242, 13746.312500, 12663.133789, 1.587083,
                                 1.507701, 0.588136, 0.588974, 0.213279, 0.263020, 0.279643,
                                 0.380302, 0.161628, 0.255969};
// 3DDFA parameters
constexpr float param_mean[62] = {0.000349, 0.000000, -0.000001, 60.167957, -0.000001, 0.000576,
                                  -0.000051, 74.278198, 0.000001, 0.000066, 0.000344, -66.671577,
                                  -346603.687500, -67468.234375, 46822.265625, -15262.046875,
                                  4350.588867, -54261.453125, -18328.033203, -1584.328857,
                                  -84566.343750, 3835.960693, -20811.361328, 38094.929688,
                                  -19967.855469, -9241.370117, -19600.714844, 13168.089844,
                                  -5259.144043, 1848.647827, -13030.662109, -2435.556152,
                                  -2254.206543, -14396.561523, -6176.329102, -25621.919922,
                                  226.394470, -6326.123535, -10867.250977, 868.465088, -5831.147949,
                                  2705.123779, -3629.417725, 2043.990112, -2446.616211, 3658.697021,
                                  -7645.989746, -6674.452637, 116.388390, 7185.597168, -1429.486816,
                                  2617.366455, -1.207096, 0.669079, -0.177608, 0.056726, 0.039678,
                                  -0.135863, -0.092240, -0.172607, -0.015804, -0.141685};
constexpr float fea_mean[19] = {31.85211958, 31.47960551, 25.94624621, 17.59138676,  9.33734005,
                                21.050208  , 23.18106602, 15.43004594,  9.29398831, 13.85440052,
                                19.86294614, 12.08337902,  8.94817309,  2.84948633,  4.69245302,
                                8.02289824, 11.18050805,  3.66598218,  4.72417443};
constexpr float fea_std[19] = {24.54039414, 22.25226275, 17.61862746, 12.53473206,  7.68812305,
                               17.04655485, 16.85963973, 12.00260959,  7.57380482, 11.02798364,
                               15.56307638,  9.63335972,  6.96759367,  2.18653714,  3.58157696,
                               6.488613  ,  8.80396665,  2.71603564,  3.69771507};

// symmetry parameters
constexpr int r_idx[19] = {17,18,19,20,21, // right eyebrow
                           36,37,38,39,40,41, // right eye
                           48,49,50,58,59,60,61,67}; //right mouse
constexpr int l_idx[19] = {26,25,24,23,22, // left eyebrow
                           45,44,43,42,47,46, // left eye
                           54,53,52,56,55,64,63,65}; // left mouse

#endif //FACETOOL_ONNXRUNTIME_INFERENCE_H
