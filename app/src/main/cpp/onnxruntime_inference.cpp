//
// Created by KYL.ai on 2021-03-17.
//

#include "onnxruntime_inference.h"
#include "logs.h"
#include "preprocess.h"
#include "postprocess.h"
#include "utils.h"

Inference::Inference(std::unique_ptr<Ort::Env>& env, const char* modelpath, int img_height,
                     int img_width, int ori_height, int ori_width, bool tddfa) : env_(std::move(env)), modelpath_(modelpath),
                                                                                 img_height_(img_height), img_width_(img_width), origH_(ori_height), origW_(ori_width) {

    LOGD("model path  %s ", modelpath_);
    LOGD("Input image height  %d ", img_height_);
    LOGD("Input image width %d ", img_width_);

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetOptimizedModelFilePath(modelpath_);
    session_  = std::make_unique<Ort::Session>(*env_.get(), modelpath_, session_options);
    printNodes();
    createInputBuffer();
    if(tddfa){
        u_base = Mat::zeros(cv::Size(1, 204), CV_32F);
        w_shape = Mat::zeros(cv::Size(40, 204), CV_32F);
        w_exp = Mat::zeros(cv::Size(10, 204), CV_32F);

        out_offset = Mat::zeros(cv::Size(68, 3), CV_32F);
        out_rotate = Mat::zeros(cv::Size(3, 3), CV_32F);
        out_shape = Mat::zeros(cv::Size(1, 40), CV_32F);
        out_expr = Mat::zeros(cv::Size(1, 10), CV_32F);
        readForTDDFA(modelpath_);
    }

    output_vers.reserve(68);
    symmetry.reserve(19);
}

void Inference::createInputBuffer()
{
    LOGD("creating input data buffer of size =  %lu", input_tensor_size);
    input_data_chw  = std::make_unique<float[]>(input_tensor_size);
    normalized = std::make_unique<float[]>(input_tensor_size);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data_chw.get(), input_tensor_size, input_node_dims.data(), input_node_dims.size());
    assert(input_tensor.IsTensor());
}

void Inference::printNodes() {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_->GetInputCount();
    input_node_names.reserve(num_input_nodes);

    LOGD("Number of input =  %zu",num_input_nodes);

    for (int i = 0; i < num_input_nodes; i++){
        char* input_name = session_->GetInputName(i, allocator);
        LOGD("Input %d : name = %s", i, input_name);
        input_node_names[i] = input_name;

        Ort::TypeInfo type_info = session_->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        LOGD("Input %d : type = %d", i, type);

        input_node_dims = tensor_info.GetShape();
        LOGD("Input %d : num_dims=%zu", i, input_node_dims.size());

        input_tensor_size = 1;
        for (int j = 0;j < input_node_dims.size(); j++)
        {
            LOGD("Input %d : dim %d = %ld",i,j,std::abs(input_node_dims[j]));
            input_tensor_size *= std::abs(input_node_dims[j]);
            input_node_dims[j] = std::abs(input_node_dims[j]);
        }
    }

    size_t num_output_nodes = session_->GetOutputCount();
    output_node_names.reserve(num_output_nodes);

    for (int i = 0; i < num_output_nodes; i++){
        char* output_name = session_->GetOutputName(i, allocator);
        LOGD("Output %d : name = %s", i, output_name);
        output_node_names[i] = output_name;

        Ort::TypeInfo type_info = session_->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        ONNXTensorElementDataType type = tensor_info.GetElementType();
        LOGD("Output %d : type = %d", i, type);

        output_node_dims = tensor_info.GetShape();
        LOGD("Output %d : num_dims=%zu", i, output_node_dims.size());

        for (int j = 0;j < output_node_dims.size(); j++)
        {
            LOGD("Output %d : dim %d = %ld ",i,j,output_node_dims[j]);
        }
    }
}

void Inference::readForTDDFA(const char* model_path) {
    std::string strFullPath(model_path);
    std::string strFilePath;
    int nFind = strFullPath.rfind("/") + 1;
    strFilePath = strFullPath.substr(0, nFind);

    std::string u_base_str = strFilePath + "u_base.txt";
    char *u_base_path = &u_base_str[0];

    std::string w_shp_str = strFilePath + "w_shp.txt";
    char *w_shp_path = &w_shp_str[0];

    std::string w_exp_str = strFilePath + "w_exp.txt";
    char *w_exp_path = &w_exp_str[0];

    read_weights(u_base_path, u_base);
    read_weights(w_shp_path, w_shape);
    read_weights(w_exp_path, w_exp);
}

std::vector<std::array<float, 4>> Inference::detection(uint8_t* pixels){

    auto start = std::chrono::high_resolution_clock::now();

    preprocess(pixels, img_height_, img_width_, 4, normalized.get(), {0.485f, 0.456f, 0.406f}, {0.229, 0.224, 0.225});
    HWCtoCHW(normalized.get(), img_height_, img_width_, 3, input_data_chw.get());
    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 2);
    assert(output_tensors.size() == 2);

    // Ort::value 타입의 모델 결과를 std::pair<float *, std::vector<int64_t>>로 캐스팅
    using DataOutputType = std::pair<float *, std::vector<int64_t>>;
    std::vector<DataOutputType> outputData;
    output_tensors.reserve(2);
    for (auto &elem : output_tensors)
    {
        outputData.emplace_back(
                std::make_pair(std::move(elem.GetTensorMutableData<float>()), elem.GetTensorTypeAndShapeInfo().GetShape()));
    }

    // output includes two tensors:
    // confidences: 1 x 17640 x 2 (2 represents 2 classes of background and face)
    // bboxes: 1 x 17640 x 4 (4 represents bbox coordinates)
    int numAnchors = outputData[0].second[1];
    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;

    for (auto indices = std::make_pair(0, 0);
         indices.first < numAnchors * 2 && indices.second < numAnchors * 4;
         indices.first += 2, indices.second += 4)
    {
        float conf = outputData[0].first[indices.first + 1];
        if (conf < 0.7)
            continue;

        float xmin = outputData[1].first[indices.second + 0] * origW_;
        float ymin = outputData[1].first[indices.second + 1] * origH_;
        float xmax = outputData[1].first[indices.second + 2] * origW_;
        float ymax = outputData[1].first[indices.second + 3] * origH_;

        bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
        scores.emplace_back(conf);
    }

    std::vector<std::array<float, 4>> afterNmsBboxes;

    predicted_speed = "";

    if (bboxes.size() == 0)
    {
        predicted_speed = "Face not include in image \n";
    }
    else
    {
        auto afterNmsIndices = nms(bboxes, scores, 0.3);
        afterNmsBboxes.reserve(afterNmsIndices.size());

        for (const auto idx : afterNmsIndices)
            afterNmsBboxes.emplace_back(bboxes[idx]);

    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);

    predicted_speed += "Face detection time: "+std::to_string(duration.count())+" ms\n";
    LOGD("Time to face detection: %" PRId64 " ms \n",duration.count());

    return afterNmsBboxes;
}

std::vector<std::array<float, 3>> Inference::TDDFA(uint8_t* pixels, int img_height, int img_width, int sx, int sy){

    auto start = std::chrono::high_resolution_clock::now();
    preprocess(pixels, img_height_, img_width_, 4, normalized.get(), {0.485f, 0.456f, 0.406f}, {0.229, 0.224, 0.225});
    HWCtoCHW(normalized.get(), img_height_, img_width_, 3, input_data_chw.get());

    auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    // Ort::value 타입의 모델 결과를 std::pair<float *, std::vector<int64_t>>로 캐스팅
    assert(output_tensors.size() == 1);
    float *f = output_tensors[0].GetTensorMutableData<float>();
    output_vers.clear();

    for (int i = 0; i < 62; i++)
        f[i] = f[i] * param_std[i] + param_mean[i];
    for (int i = 0; i < 68; i++)
    {
        out_offset.at<float>(0, i) = f[3];
        out_offset.at<float>(1, i) = f[7];
        out_offset.at<float>(2, i) = f[11];
    }
    out_rotate.at<float>(0, 0) = f[0];
    out_rotate.at<float>(0, 1) = f[1];
    out_rotate.at<float>(0, 2) = f[2];
    out_rotate.at<float>(1, 0) = f[4];
    out_rotate.at<float>(1, 1) = f[5];
    out_rotate.at<float>(1, 2) = f[6];
    out_rotate.at<float>(2, 0) = f[8];
    out_rotate.at<float>(2, 1) = f[9];
    out_rotate.at<float>(2, 2) = f[10];
    for (int i = 0; i < 40; i++)
        out_shape.at<float>(i, 0) = f[12 + i];
    for (int i = 0; i < 10; i++)
        out_expr.at<float>(i, 0) = f[52 + i];
    Mat mat_mul = Mat::zeros(cv::Size(1, 204), CV_32F);
    mat_mul = u_base + w_shape * out_shape + w_exp * out_expr;
    out_offset = out_rotate * mat_mul.reshape(1, 68).t() + out_offset;
    double min, max;
    minMaxLoc(out_offset, &min, &max);
    float scale_x = img_width / (float)120.0;
    float scale_y = img_height / (float)120.0;

    for (int i = 0; i < 68; i++)
    {
        std::array<float, 3> temp = {(out_offset.at<float>(0, i)) * scale_x + sx,
                                     (120 - out_offset.at<float>(1, i)) * scale_y + sy,
                                     (float)((out_offset.at<float>(2, i)) * ((scale_x + scale_y) / 2.0) - min)*-1
        };
        output_vers.emplace_back(temp);
    }

    predicted_speed = "";
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::milliseconds>(stop - start);

    predicted_speed += "Facial landmark time: "+std::to_string(duration.count())+" ms\n";
    LOGD("Time to TDDFA: %" PRId64 " ms \n",duration.count());

    return output_vers;
}

Point Inference::gaze(Mat pixels, int width, int height){

    auto start = std::chrono::high_resolution_clock::now();

    Mat eye_grey;
    cvtColor(pixels, eye_grey, COLOR_RGB2GRAY);
    GaussianBlur(eye_grey,eye_grey,Size(5,5),0);

    // Calculate image gradients
    Mat grad_x, grad_y;
    Sobel(eye_grey, grad_x, CV_32F, 1, 0, 5);
    Sobel(eye_grey, grad_y, CV_32F, 0, 1, 5);

    // Get magnitudes of gradients, and calculate thresh
    Mat mags;
    Scalar mean, stddev;
    magnitude(grad_x, grad_y, mags);
    meanStdDev(mags, mean, stddev);
    int mag_thresh = stddev.val[0] / 2 + mean.val[0];

    // Threshold out gradients with mags which are too low
    grad_x.setTo(0, mags < mag_thresh);
    grad_y.setTo(0, mags < mag_thresh);

    // Normalize gradients
    grad_x = grad_x / (mags+1); // (+1 is hack to guard against div by 0)
    grad_y = grad_y / (mags+1);

    // Initialize 1d vectors of x and y indicies of Mat
    std::vector<int> x_inds_vec, y_inds_vec;
    for(int i = 0; i < eye_grey.size().width; i++)
        x_inds_vec.push_back(i);
    for(int i = 0; i < eye_grey.size().height; i++)
        y_inds_vec.push_back(i);

    // Repeat vectors to form indices Mats
    Mat x_inds(x_inds_vec), y_inds(y_inds_vec);
    x_inds = repeat(x_inds.t(), eye_grey.size().height, 1);
    y_inds = repeat(y_inds, 1, eye_grey.size().width);
    x_inds.convertTo(x_inds, CV_32F);	// Has to be float for arith. with dx, dy
    y_inds.convertTo(y_inds, CV_32F);

    // Set-up Mats for main loop
    Mat ones = Mat::ones(x_inds.rows, x_inds.cols, CV_32F);	// for re-use with creating normalized disp. vecs
    Mat darkness_weights = (255 - eye_grey) / 100;
    Mat accumulator = Mat::zeros(eye_grey.size(), CV_32F);
    Mat diffs, dx, dy;

    // Loop over all pixels, testing each as a possible center
    for(int y = 0; y < eye_grey.rows; ++y) {

        // Get pointers for each row
        float* grd_x_p = grad_x.ptr<float>(y);
        float* grd_y_p = grad_y.ptr<float>(y);
        uchar* d_w_p = darkness_weights.ptr<uchar>(y);

        for(int x = 0; x < eye_grey.cols; ++x) {

            // Deref and increment pointers
            float grad_x_val = *grd_x_p++;
            float grad_y_val = *grd_y_p++;

            // Skip if no gradient
            if(grad_x_val == 0 && grad_y_val == 0)
                continue;

            dx = ones * x - x_inds;
            dy = ones * y - y_inds;

            magnitude(dx, dy, mags);
            dx = dx / mags;
            dy = dy / mags;

            diffs = (dx * grad_x_val + dy * grad_y_val) * *d_w_p++;
            diffs.setTo(0, diffs < 0);

            accumulator = accumulator + diffs;
        }
    }

    // Normalize and convert accumulator
    accumulator = accumulator / eye_grey.total();
    normalize(accumulator, accumulator, 0, 255, NORM_MINMAX);
    accumulator.convertTo(accumulator, CV_8U);

    // Find position of max value in small-size centermap
    Point maxLoc;
    minMaxLoc(accumulator, NULL, NULL, NULL, &maxLoc);

    return maxLoc;
}


std::string Inference::getPredictedlabels()
{
    return predicted_speed;
}

Inference::~Inference(){
    delete modelpath_;
    delete labelfilepath_;
}


