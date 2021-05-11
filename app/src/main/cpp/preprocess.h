//
// Created by KYL.ai on 2021-03-17.
//

#ifndef FACETOOL_PREPROCESS_H
#define FACETOOL_PREPROCESS_H

/**
 * Method to convert HWC to CHW format
 * @param input : normalized image pointer with HWC format
 * @param height : height of image
 * @param width : width of image
 * @param channels : channels of image
 * @param output : normalized image pointer with CHW format
 */
void HWCtoCHW(float* input,int height, int width, int channels, float* output)
{   /// image[i,j,c] = img[(width*i + j )*channels +c] - > HWC : image[c,i,j] = img[(height*c +i)*width + j] - > CHW
    for(int c = 0; c < channels; c++)
    {
        for(int i = 0; i<height; i++ )
        {
            for(int j=0; j<width; j++)
            {
                output[ (height*c +i)*width + j ] = input[ (width*i + j )*channels +c];
            }
        }
    }

    return;
}

/**
 *  method to skip Alpha channel and normalize the image
 * @param input : RGBA img pointer from bitmap
 * @param height : height of img
 * @param width : width of img
 * @param channels : channels of img (4 RBA)
 * @param output : normalized imag pointer
 * @param mean : meean values
 * @param std : std values;
 */
void preprocess(uint8_t* input, int height, int width, int channels, float* output, std::vector<float> mean, std::vector<float> std)
{
    int channel_out = 3;
    for(int i=0; i<height; i++)
    {
        for(int j=0; j<width; j++)
        {
            for(int c=0;c <channel_out; c++) //skip alpha channae;
            {
                float normalize = input[(i*width +j)*channels + c] / 255.0f;
                output[(i*width +j)*channel_out + c] = ( normalize - mean[c])/std[c];

            }
        }
    }

    return;
}

void gray_preprocess(float *dst, const unsigned char *src, int height, int width)
{
    for (int c = 0; c < 1; ++c)
    {
        for (int i = 0; i < height; ++i)
        {
            for (int j = 0; j < width; ++j)
            {
                dst[c * height * width + i * width + j] =
                        (src[i * width * 1 + j * 1 + c] - (float)127.) / (float)128.;
            }
        }
    }

    return;
}

#endif //FACETOOL_PREPROCESS_H
