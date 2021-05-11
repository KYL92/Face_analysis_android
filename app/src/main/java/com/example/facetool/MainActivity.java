package com.example.facetool;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.CaptureResult;
import android.hardware.camera2.TotalCaptureResult;
import android.os.Build;
import android.os.CountDownTimer;
import android.os.Handler;
import android.os.HandlerThread;
import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.Vector;

public class MainActivity extends AppCompatActivity {

    private int CAMERA = 10;
    // Transfer the image received from the camera to mTextureView.
    // mSurfaceView is a space for drawing face positions, feature points, and graphs on the screen.
    // Display analysis result by drawing mSurfaceView on mTextureView
    private TextureView mTextureView;
    private TextView mTextView;
    private SurfaceView mSurfaceView;
    private SurfaceHolder mHolder;

    static String TAG = "onnxruntime_inference";
    private String mCameraId;
    private HandlerThread mBackgroundThread;
    private Handler mBackgroundHandler;
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mCaptureSession;
    private CaptureRequest.Builder captureRequestBuilder;
    private CaptureRequest captureRequest;
    private final Object lock = new Object();
    private boolean runFaceToolkit = false;
    private int ori_height = 1080; // Smartphone FHD resolution setting is essential
    private int ori_width = 1080; // Smartphone FHD resolution setting is essential
    private int mheigth = 0;
    private int mwidth = 0;
    // Inference is a class for creating and executing deep learning models.
    private Inference face_detector = null; // Facebox detection AI model
    private Inference TDDFA_model = null; // 3D facial landmark point detection AI model
    // Receive an image of mTextureView and save it as a Bitmap
    private Bitmap bitmap = null;
    private Bitmap bitmap_ = null;
    private Bitmap resizedbitmap = null;
    // Paint and Rect for display of face analysis results in msufaceview
    private Paint paint;

    // Variable that receives model results
    private Vector<Vector<Float>> output_boxs = null; // Variable that Face area box coordinates
    private Vector<Vector<Float>> output_vers = null; // Variable that facial landmark point (3D)
    private int frame_number = 0;
    private int time_count = 0;

    // mTextureView에 카메라 이미지 띄우기 위한 작업
    private final TextureView.SurfaceTextureListener textureListener =
            new TextureView.SurfaceTextureListener() {

                @Override
                public void onSurfaceTextureAvailable(SurfaceTexture texture, int width, int height) {
                    openCamera(width, height);
                }
                @Override
                public void onSurfaceTextureSizeChanged(SurfaceTexture texture, int width, int height) {
                }
                @Override
                public boolean onSurfaceTextureDestroyed(SurfaceTexture texture) {
                    return true;
                }
                @Override
                public void onSurfaceTextureUpdated(SurfaceTexture texture) {
                    frame_number = frame_number +1;
                    if(frame_number>30){
                        frame_number = 0;
                    }
                }
            };

    // 카메라 설정
    private final CameraDevice.StateCallback mStateCallback =
            new CameraDevice.StateCallback() {

                @Override
                public void onOpened(@NonNull CameraDevice cameraDevice) {
                    // This method is called when the camera is opened.  We start camera preview here.
                    //cameraOpenCloseLock.release();
                    Log.d(TAG, "Camera Opened");
                    mCameraDevice = cameraDevice;
                    createPreviewSession();

                }

                @Override
                public void onDisconnected(@NonNull CameraDevice currentCameraDevice) {
                    //cameraOpenCloseLock.release();
                    Log.d(TAG, "Camera Disconnected");
                    mCameraDevice.close();
                    mCameraDevice = null;
                }

                @Override
                public void onError(@NonNull CameraDevice currentCameraDevice, int error) {
                    //cameraOpenCloseLock.release();
                    Log.d(TAG, "Camera Error");
                    mCameraDevice.close();
                    mCameraDevice = null;
                }
            };

    // 카메라 설정
    private CameraCaptureSession.CaptureCallback captureCallback =
            new CameraCaptureSession.CaptureCallback() {

                @Override
                public void onCaptureProgressed(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull CaptureResult partialResult) {
                }

                @Override
                public void onCaptureCompleted(
                        @NonNull CameraCaptureSession session,
                        @NonNull CaptureRequest request,
                        @NonNull TotalCaptureResult result) {
                }
            };

    // Inference creation for initial deep learning inference, model loading, and other settings
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // FLAG_KEEP_SCREEN_ON option set and converted to full screen so that the screen does not automatically turn off while the app is running
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            final WindowInsetsController insetsController = getWindow().getInsetsController();
            if (insetsController != null) {
                insetsController.hide(WindowInsets.Type.statusBars());
            }
        } else {
            getWindow().setFlags(
                    WindowManager.LayoutParams.FLAG_FULLSCREEN,
                    WindowManager.LayoutParams.FLAG_FULLSCREEN
            );
        }
        // Calling layouts and setting paint for display of feature points and face analysis results
        mTextureView = (TextureView) findViewById(R.id.texture_view);
        mSurfaceView = (SurfaceView) findViewById(R.id.surfaceView);
        mTextView = (TextView) findViewById(R.id.text_view);
        mSurfaceView.setZOrderOnTop(true);
        mHolder = mSurfaceView.getHolder();
        mHolder.setFormat(PixelFormat.TRANSPARENT);
        mTextView.setTextColor(Color.RED);
        paint = new Paint();
        paint.setColor(Color.GREEN);
        paint.setStrokeWidth(5);
        paint.setStyle(Paint.Style.STROKE);
        // Receives the absolute path of the model file in the assets folder through the assetFilePath function
        String face_detector_path = new File(assetFilePath(this, "facedetector.onnx")).getAbsolutePath();
        String TDDFA_path = new File(assetFilePath(this, "TDDFA.onnx")).getAbsolutePath();
        // In the case of TDDFA.onnx, there are separate parameters for 3D modeling, and check and verify their paths below.
        assetFilePath(this, "u_base.txt");
        assetFilePath(this, "w_exp.txt");
        assetFilePath(this, "w_shp.txt");
        // Inference instance creation of deep learning model (a face detection model is created and defined through JNI.)
        // public Inference(String model_path, // 모델 절대경로
        //                  int img_height, // 모델 입력해상도 높이
        //                  int img_width, // 모델 입력해상도 넓이
        //                  int ori_height, // 화면 해상도 높이
        //                  int ori_width, // 화면 해상도 넓이
        //                  boolean tddfa // 얼굴특징점모델 추론 여부
        face_detector = new Inference(face_detector_path, 240, 320, ori_height, ori_width, false);
        TDDFA_model = new Inference(TDDFA_path, 120, 120, ori_height, ori_width, true);
    }


    @Override
    protected void onResume() {
        super.onResume();
        // hideSystemUI(); //hide UI
        Log.d(TAG, "on Resume");
        if (mTextureView.isAvailable()) {
            Log.d(TAG, "mTexture is available");
            openCamera(mwidth, mheigth);
        } else
            mTextureView.setSurfaceTextureListener(textureListener);
        openBackgroundThread();//start the backgroundthread
    }

    @Override
    protected void onPause() {
        closeCamera();
        closeBackgroundThread();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "on  Destory");
        if (face_detector != null)
            face_detector.delete();
        if (bitmap != null)
            bitmap.recycle();
        if (resizedbitmap != null)
            resizedbitmap.recycle();
    }

    // This app uses only the front camera. OpenCamera function for front camera index and camera setting
    private static final int CAMERA_INDEX = 1;
    private void openCamera(int width, int height) {

        if (Build.VERSION.SDK_INT >= 23)
            if (!askForPermission(Manifest.permission.CAMERA, CAMERA)) //ask for camera permission
                return;
        try {
            CameraManager cameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
            mCameraId = cameraManager.getCameraIdList()[CAMERA_INDEX];

            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                return;
            }
            cameraManager.openCamera(mCameraId, mStateCallback, mBackgroundHandler);
            mwidth = mTextureView.getWidth();
            mheigth = mTextureView.getHeight();
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
    // 카메라 권한 받는 함수
    private boolean askForPermission(String permission, Integer requestCode) {
        if (ContextCompat.checkSelfPermission(MainActivity.this, permission) != PackageManager.PERMISSION_GRANTED) {
            if (ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this, permission)) {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
                return false;
            } else {
                ActivityCompat.requestPermissions(MainActivity.this, new String[]{permission}, requestCode);
            }
        } else {
            Log.v(TAG, "permission  is already granted.");
            return true;
        }
        return false;
    }
    // 카메라 권한 받는 함수
    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (ActivityCompat.checkSelfPermission(this, permissions[0]) == PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show();
        } else{

            Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show();
        }
    }
    // 카메라 설정
    private void createPreviewSession()
    {
        try {
            SurfaceTexture surfaceTexture = mTextureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(1440,1080);
            Surface previewSurface = new Surface(surfaceTexture);

            captureRequestBuilder = mCameraDevice.createCaptureRequest(mCameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(previewSurface);

            mCameraDevice.createCaptureSession(Collections.singletonList(previewSurface),

                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession cameraCaptureSession) {
                            if(mCameraDevice == null)
                            {
                                return;
                            }

                            try {
                                mCaptureSession = cameraCaptureSession;
                                captureRequestBuilder.set(
                                        CaptureRequest.CONTROL_AF_MODE,
                                        CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                captureRequest = captureRequestBuilder.build();

                                mCaptureSession.setRepeatingRequest(captureRequest,captureCallback,mBackgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }

                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession cameraCaptureSession) {

                        }
                    },mBackgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
            Log.e(TAG, "Failed to preview Camera", e);
        }
    }

    // FaceToolkit 프로세스 수행 시간 측정을 위한 변수
    private long startTime = 0;
    private long endTime = 0;
    // 모델 입력 종횡비 4:3을 맞추기 위한 변수
    private int aspect_y = 0;
    private int w = 0;
    private int h = 0;
    private int x = 0;
    private int y = 0;
    private Canvas canvas = null;
    private String output_text;
    private String detection_text;

    // 딥러닝 모델 실행 함수, BackgroundThread로 동작한다.
    private void FaceToolkit() {
        if (face_detector == null) {
            Log.e(TAG,"Uninitialized inference or invalid context.");
        }
        if (mCameraDevice == null) {
            Log.e(TAG,"Uninitialized mCameraDevice or invalid context.");
            return;
        }
        // mTextureView의 Bitmap을 받아온다. Bitmap 이미지는 딥러닝 모델 입력으로 사용됨
        bitmap = mTextureView.getBitmap();
        // The front camera needs to convert the input image to the model's input ratio (320x240) with a longer aspect ratio.
        aspect_y = bitmap.getHeight()/2 - bitmap.getWidth()/2;
        Log.v(TAG,"bitmap.getWidth() "+ bitmap.getWidth() + "bitmap.getHeight" + bitmap.getHeight());

        if (bitmap.getWidth() <= bitmap.getHeight()) { //portrait
            // crop and resize using android.graphics.Bitmap 해상도 변환을 위한 이미지 잘라내기
            bitmap_ = Bitmap.createBitmap(bitmap, 0, aspect_y, bitmap.getWidth(), bitmap.getWidth()); // center crop
            // 얼굴영역 검출 모델(face_detector)을 위한 이미지 리사이즈
            resizedbitmap = Bitmap.createScaledBitmap(bitmap_, 320, 240, true);// resize
            // FaceToolkit 처리속도 측정
            startTime = SystemClock.uptimeMillis();
            // first face analysis, face detection
            if (frame_number%10 == 0){
                // Face box detection model execution One or more face region coordinates are returned in output_boxs.
                // ex) If there are 4 faces in the image, the model contains 4 (start x, start y, end x, end y) box coordinates.
                //     In other words, output_boxs = [[sx1, sy1, ex1, ey1], [sx2, sy2, ex2, ey2], [sx3, sy3, ex3, ey3], [sx4, sy4, ex4, ey4]]
                output_boxs = face_detector.detection(resizedbitmap);
                detection_text = face_detector.getfps();
            }
            // mSurfaceView에 그리기 위해 canvas 받기
            canvas = mHolder.lockCanvas();

            if (canvas == null) {
                Log.e(TAG, "Cannot draw onto the canvas as it's null");
            } else {
                // mSurfaceView 배경을 투명으로 설정
                canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

                if (output_boxs.size() == 1){
                    // processing face analysis for each face box
                    for (Vector<Float> box : output_boxs) {
                        // convert box size to model's input size
                        w = Math.round(box.get(2) - box.get(0));
                        h = Math.round((box.get(3) - box.get(1)));
                        x = Math.round(box.get(0)-w/8);
                        y = Math.round(box.get(1) + aspect_y + h/10);
                        w = w + w/4;
                        h = h - h/20;
                        // exception Handling
                        if ( x<=0 || y<=0 || w<=0 || h<=0 || x + w >= bitmap.getWidth() || y + h >= bitmap.getHeight() ) break;
                        // crop and resized
                        bitmap_ = Bitmap.createBitmap(bitmap, x, y, w, h);
                        // Facial landmark point model (TDDFA) and image resizing for facial expression recognition
                        resizedbitmap = Bitmap.createScaledBitmap(bitmap_, 120, 120, true);// resize
                        // run TDDFA model for facial landmark (얼굴 특징점 검출모델)
                        // public Vector<Vector<Float>> TDDFA(Bitmap input, 비트맵 입력 이미지
                        //                                    int img_height, 얼굴영역 높이
                        //                                    int img_width, 얼굴영역 넓이
                        //                                    int sx, 얼굴영역 시작점 X좌표
                        //                                    int sy) 얼굴영역 시작점 Y좌표
                        output_vers = TDDFA_model.TDDFA(resizedbitmap, h, w, x, y);

                        // draw face box
                        canvas.drawRect(x, y, x + w, y + h, paint);
                        // draw landmark
                        paint.setStrokeWidth(5);
                        for (Vector<Float> vertex : output_vers) {
                            canvas.drawCircle(vertex.get(0), vertex.get(1), 5, paint);
                        }

                        // 모델별 추론속도 확인을 위한 text 작성
                        output_text = detection_text + TDDFA_model.getfps();
                        paint.setColor(Color.GREEN);

                    }
                }
                // 얼굴분석결과 DP 후 canvas Post
                mHolder.unlockCanvasAndPost(canvas);
            }
            // 실행완료 시간 확인 및 DP
            endTime = SystemClock.uptimeMillis();
            Log.e(TAG, "inference " + Long.toString(endTime - startTime) + " in mills ");

            if (output_boxs.size() == 1){
                output_text = "Turnaround: " + Long.toString(Math.round(1000/(endTime - startTime))) + " FPS "
                        + Long.toString(time_count) +"\n" + output_text;
            }
            else{
                output_text = "KYL.ai\n";
            }

            // UI에 text 보여주기
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mTextView.setText(output_text);
                }
            });
        }
        else //land scape 는 구현 안함
        {
            Log.e(TAG, "landscape");
            bitmap = Bitmap.createBitmap( bitmap, 0,  aspect_y,  bitmap.getWidth(),  bitmap.getWidth()); //center crop
            resizedbitmap = Bitmap.createScaledBitmap(bitmap, 320, 240, true);//resize
            output_boxs = face_detector.detection(resizedbitmap);
            output_text = face_detector.getfps();
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    mTextView.setText(output_text);
                }
            });
        }
    }

    // 백그라운드 쓰레드를 사용하여 FaceToolkit 함수를 주기적으로 실행
    private Runnable periodicEvaluation;
    {
        periodicEvaluation = new Runnable() {
            @Override
            public void run() {
                synchronized (lock) {
                    if (runFaceToolkit) {
                        FaceToolkit();
                        Log.d(TAG, "peroidic evaluation in background thread");
                    }
                }
                mBackgroundHandler.post(periodicEvaluation);//if load
            }
        };
    }

    // 백그라운드 쓰레드 실행
    private void openBackgroundThread()
    {
        mBackgroundThread = new HandlerThread("camera background thread");
        mBackgroundThread.start();
        mBackgroundHandler = new Handler( mBackgroundThread.getLooper());
        synchronized (lock) {
            runFaceToolkit = true;
        }
        mBackgroundHandler.post(periodicEvaluation);
    }

    // 백그라운드 쓰레드 종료
    private void closeBackgroundThread()
    {
        mBackgroundThread.quitSafely();
        try {
            mBackgroundThread.join();
            mBackgroundThread = null;
            mBackgroundHandler = null;
            synchronized (lock) {
                runFaceToolkit = false;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    // 카메라 종료
    private void closeCamera()
    {
        if (null != mCaptureSession) {
            mCaptureSession.close();
            mCaptureSession = null;
        }
        if (mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }
    }

    // Check the model and parameter files in the asset folder and return the absolute path
    public static String assetFilePath(Context context, String assetName) {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        } catch (IOException e) {
            Log.e(TAG, "Error process asset " + assetName + " to file path");
        }
        return null;
    }

}
