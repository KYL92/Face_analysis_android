package com.example.facetool;

import android.graphics.Bitmap;

import java.util.Vector;

public class Inference {
    // Inference 인스턴스를 관리 및 선택을 용이하게하기 위한 주소값 저장 변수
    private long selfAddr;

    static {
        System.loadLibrary("native-lib");
    }

    /**
     * makes jni call to create c++ reference
     */
    // 모델 로드 및 생성을 위한 Inference 생성자
    // model_path: 딥러닝 모델의 절대 경로 assets 폴더에 존재
    // img_height, img_width: 입력 이미지의 해상도
    // ori_height, ori_width: 핸드폰 화면의 해상도
    // tddfa는 얼굴특징점 모델의 경우 3D 모델링을 위한 파라미터 파일이 필요함으로 옵션 설정
    public Inference(String model_path, int img_height, int img_width, int ori_height, int ori_width, boolean tddfa)
    {
        selfAddr = newSelf(model_path, img_height, img_width, ori_height, ori_width, tddfa); //jni call to create c++ reference and returns address
    }

    /**
     * makes jni call to delete c++ reference
     */
    public void delete()
    {
        deleteSelf(selfAddr);//jni call to delete c++ reference
        selfAddr = 0;//set address to 0
    }

    @Override
    protected void finalize() throws Throwable {
        delete();
    }

    /**
     * return address of c++ reference
     */
    // Inference 인스턴스의 주소를 반환
    public long getselfAddr() {

        return selfAddr; //return address
    }

    /**
     * //makes jni call to proces frames
     */
    // 얼굴영역검출모델 실행
    public Vector<Vector<Float>> detection(Bitmap input) {
        return detection(selfAddr, input);//jni call to proces frames
    }

    /**
     * //makes jni call to proces frames
     */
    // 얼굴특징점검출모델 실행
    public Vector<Vector<Float>> TDDFA(Bitmap input, int img_height, int img_width, int sx, int sy) {
        return TDDFA(selfAddr, input, img_height, img_width, sx, sy);//jni call to proces frames
    }
    // 얼굴특징점검출모델 실행
    public Vector<Float> gaze(Bitmap input, int img_height, int img_width) {
        return gaze(selfAddr, input,img_height,img_width);//jni call to proces frames
    }
    // gaze svm
    public Vector<Float> gazesvm(Vector<Float> data) {

        float[] data_arr = new float[data.size()];
        int i =0;
        for(Float item : data){
            data_arr[i] = item;
            i++;
        }

        return gazesvm(selfAddr, data_arr);//jni call to proces frames
    }

    /**
     * //makes jni call to proces frames
     */
    public String getfps() {
        return getfps(selfAddr);//jni call to proces frames
    }
    private static native long newSelf(String model_path, int img_height, int img_width,
                                       int ori_height, int ori_width, boolean tddfa);
    private static native void deleteSelf(long selfAddr);
    private static native Vector<Vector<Float>> detection(long selfAddr, Bitmap inbitmap);
    private static native Vector<Vector<Float>> TDDFA(long selfAddr, Bitmap inbitmap,
                                                      int img_height, int img_width, int sx, int sy);
    private static native Vector<Float> gaze(long selfAddr, Bitmap inbitmap, int img_height, int img_width);
    private static native Vector<Float> gazesvm(long selfAddr, float[] data);
    private static native String getfps(long selfAddr);
}
