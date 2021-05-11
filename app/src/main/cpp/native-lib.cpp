//
// Created by KYL.ai on 2021-03-17.
//

#include <jni.h>
#include <string>
#include <android/bitmap.h>
#include "onnxruntime_inference.h"
#include "logs.h"

#define ASSERT(status, ret)     if (!(status)) { return ret; }
#define ASSERT_FALSE(status)    ASSERT(status, false)

bool BitmapToMatrix(JNIEnv * env, jobject obj_bitmap, Mat & matrix) {
    void * bitmapPixels;                                            // Save picture pixel data
    AndroidBitmapInfo bitmapInfo;                                   // Save picture parameters

    ASSERT_FALSE( AndroidBitmap_getInfo(env, obj_bitmap, &bitmapInfo) >= 0);        // Get picture parameters
    ASSERT_FALSE( bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888
                  || bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGB_565 );          // Only ARGB? 8888 and RGB? 565 are supported
    ASSERT_FALSE( AndroidBitmap_lockPixels(env, obj_bitmap, &bitmapPixels) >= 0 );  // Get picture pixels (lock memory block)
    ASSERT_FALSE( bitmapPixels );

    if (bitmapInfo.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC4, bitmapPixels);    // Establish temporary mat
        tmp.copyTo(matrix);                                                         // Copy to target matrix
    } else {
        Mat tmp(bitmapInfo.height, bitmapInfo.width, CV_8UC2, bitmapPixels);
        cvtColor(tmp, matrix, COLOR_BGR5652RGB);
    }
    AndroidBitmap_unlockPixels(env, obj_bitmap);            // Unlock
    return true;
}

extern "C"
JNIEXPORT jlong JNICALL
Java_com_example_facetool_Inference_newSelf(JNIEnv *env, jclass clazz, jstring model_path,
                                            jint img_height, jint img_width, jint ori_height,
                                            jint ori_width, jboolean tddfa) {
    std::unique_ptr<Ort::Env> environment(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE,"test"));
    const char *model_path_ch = env->GetStringUTFChars(model_path, 0);
    Inference *self = new Inference(environment, model_path_ch, img_height, img_width, ori_height, ori_width, tddfa);
    return (jlong) self;
}extern "C"
JNIEXPORT void JNICALL
Java_com_example_facetool_Inference_deleteSelf(JNIEnv *env, jclass clazz, jlong selfAddr) {
    if (selfAddr != 0) {
        Inference *self = (Inference *) selfAddr;
        LOGE("deleted c++ object");
        delete self;
    }
}extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_facetool_Inference_detection(JNIEnv *env, jclass clazz, jlong selfAddr,
                                              jobject inputbitmap) {
    if (selfAddr != 0) {
        AndroidBitmapInfo info;
        Inference *self = (Inference *) selfAddr;
        uint8_t *inputpixel;
        int ret;
        if ((ret = AndroidBitmap_getInfo(env, inputbitmap, &info)) < 0) {
            LOGE("Input AndroidBitmap_getInfo() failed ! error=%d", ret);
            return env->NewStringUTF("none");
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGE("InputBitmap format is not RGBA_8888 !");
            return env->NewStringUTF("none");
        }
        if ((ret = AndroidBitmap_lockPixels(env, inputbitmap, (void **) &inputpixel)) < 0) {
            LOGE("Input AndroidBitmap_lockPxels() failed ! error=%d", ret);
        }
        LOGD("bitmap width %d , bitmap heigth %d, bitmap stride %d", info.width, info.height, info.stride);
        AndroidBitmap_unlockPixels(env, inputbitmap);
        std::vector<std::array<float, 4>> res;

        // run face detection model
        res = self->detection(inputpixel);

        // java object for return
        jclass vectorClass = env->FindClass("java/util/Vector");
        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID mid = env->GetMethodID(vectorClass, "<init>", "()V");
        jmethodID addMethodID = env->GetMethodID(vectorClass, "add", "(Ljava/lang/Object;)Z");
        // result vector
        jobject result = env->NewObject(vectorClass, mid);
        for(std::array<float, 4> row : res) {
            // Inner vector
            jobject innerVector = env->NewObject(vectorClass, mid);
            for(float f : row) {
                jmethodID floatConstructorID = env->GetMethodID(floatClass, "<init>", "(F)V");
                // Now, we have object created by Float(f)
                jobject floatValue = env->NewObject(floatClass, floatConstructorID, f);
                env->CallBooleanMethod(innerVector, addMethodID, floatValue);
            }
            env->CallBooleanMethod(result, addMethodID, innerVector);
        }

        env->DeleteLocalRef(vectorClass);
        env->DeleteLocalRef(floatClass);

        return result;
    }
    return 0;
}extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_facetool_Inference_TDDFA(JNIEnv *env, jclass clazz, jlong selfAddr,
                                          jobject inputbitmap, jint img_height, jint img_width,
                                          jint sx, jint sy) {
    if (selfAddr != 0) {
        AndroidBitmapInfo info;
        Inference *self = (Inference *) selfAddr;
        uint8_t *inputpixel;
        int ret;
        if ((ret = AndroidBitmap_getInfo(env, inputbitmap, &info)) < 0) {
            LOGE("Input AndroidBitmap_getInfo() failed ! error=%d", ret);
            return env->NewStringUTF("none");
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOGE("InputBitmap format is not RGBA_8888 !");
            return env->NewStringUTF("none");
        }
        if ((ret = AndroidBitmap_lockPixels(env, inputbitmap, (void **) &inputpixel)) < 0) {
            LOGE("Input AndroidBitmap_lockPxels() failed ! error=%d", ret);
        }
        LOGD("bitmap width %d , bitmap heigth %d, bitmap stride %d", info.width, info.height, info.stride);
        AndroidBitmap_unlockPixels(env, inputbitmap);
        std::vector<std::array<float, 3>> res;

        // run TDDFA model
        res = self->TDDFA(inputpixel, img_height, img_width, sx, sy);

        // java object for return
        jclass vectorClass = env->FindClass("java/util/Vector");
        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID mid = env->GetMethodID(vectorClass, "<init>", "()V");
        jmethodID addMethodID = env->GetMethodID(vectorClass, "add", "(Ljava/lang/Object;)Z");

        // result vector
        jobject result = env->NewObject(vectorClass, mid);
        for(std::array<float, 3> row : res) {
            // Inner vector
            jobject innerVector = env->NewObject(vectorClass, mid);
            for(float f : row) {
                jmethodID floatConstructorID = env->GetMethodID(floatClass, "<init>", "(F)V");
                // Now, we have object created by Float(f)
                jobject floatValue = env->NewObject(floatClass, floatConstructorID, f);
                env->CallBooleanMethod(innerVector, addMethodID, floatValue);
            }
            env->CallBooleanMethod(result, addMethodID, innerVector);
        }

        env->DeleteLocalRef(vectorClass);
        env->DeleteLocalRef(floatClass);

        return result;
    }
    return 0;
}extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_facetool_Inference_getfps(JNIEnv *env, jclass clazz, jlong selfAddr) {
    if (selfAddr != 0) {
        Inference *self = (Inference *) selfAddr;
        jstring label = env->NewStringUTF(self->getPredictedlabels().c_str());
        return label;
    }
    return env->NewStringUTF("None");;
}extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_facetool_Inference_gaze(JNIEnv *env, jclass clazz, jlong selfAddr,
                                         jobject inputbitmap, jint img_height, jint img_width) {

    Mat matBitmap;
    bool ret = BitmapToMatrix(env, inputbitmap, matBitmap);// Bitmap to cv::Mat
    if (ret == false) {
        return 0;
    }
    if (selfAddr != 0) {
        Inference *self = (Inference *) selfAddr;
        Point res = self->gaze(matBitmap, img_height, img_width);
        Mat skin;
        //first convert our RGB image to YCrCb
        cv::cvtColor(matBitmap,skin,cv::COLOR_RGB2YCrCb);
        //filter the image in YCrCb color space
        cv::inRange(skin,cv::Scalar(0,133,77),cv::Scalar(235,173,127),skin);
        float skin_ratio = (float(sum(skin)[0])/255.0f)/(img_height*img_width);

        // java object for return
        jclass vectorClass = env->FindClass("java/util/Vector");
        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID mid = env->GetMethodID(vectorClass, "<init>", "()V");
        jmethodID addMethodID = env->GetMethodID(vectorClass, "add", "(Ljava/lang/Object;)Z");
        jobject result = env->NewObject(vectorClass, mid);

        jmethodID floatConstructorID = env->GetMethodID(floatClass, "<init>", "(F)V");
        // Now, we have object created by Float(f)
        jobject floatValue = env->NewObject(floatClass, floatConstructorID, float(res.x));
        env->CallBooleanMethod(result, addMethodID, floatValue);
        floatValue = env->NewObject(floatClass, floatConstructorID, float(res.y));
        env->CallBooleanMethod(result, addMethodID, floatValue);
//        floatValue = env->NewObject(floatClass, floatConstructorID, log2(2.0f - skin_ratio));
        floatValue = env->NewObject(floatClass, floatConstructorID, (1.0f-skin_ratio));
        env->CallBooleanMethod(result, addMethodID, floatValue);

        env->DeleteLocalRef(vectorClass);
        env->DeleteLocalRef(floatClass);
        env->DeleteLocalRef(floatValue);

        return result;
    }
    return 0;
}