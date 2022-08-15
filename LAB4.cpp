#include <opencv2\opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

Mat image, gos_znak, _symbol;
Mat DFT_convolute(Mat img, vector<vector<int8_t>> filter, Mat Output);
Mat DFT_Correlation(Mat img1, Mat img2);
Mat return_low_frequence(Mat spectr);
Mat return_high_frequence(Mat spectr);

Mat make_spectr_better(Mat src) {
    Mat img = src.clone();
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    Mat q0(img, Rect(0, 0, cx, cy));
    Mat q1(img, Rect(cx, 0, cx, cy));
    Mat q2(img, Rect(0, cy, cx, cy));
    Mat q3(img, Rect(cx, cy, cx, cy));

    Mat temp;
    q0.copyTo(temp);
    q3.copyTo(q0);
    temp.copyTo(q3);

    q1.copyTo(temp);
    q2.copyTo(q1);
    temp.copyTo(q2);

    return img;
}

Mat return_img_spectr(Mat img) {
    Size dftSize;
    dftSize.width = getOptimalDFTSize(img.cols);
    dftSize.height = getOptimalDFTSize(img.rows);
    Mat canvas_image(dftSize, img.type(), Scalar::all(0));
    Mat insert_region_image(canvas_image, Rect(0, 0, img.cols, img.rows));
    img.copyTo(insert_region_image);
    dft(canvas_image, canvas_image, DFT_COMPLEX_OUTPUT);
    return canvas_image;
}

Mat open_image(string image_name) {
    Mat img = imread(image_name, IMREAD_GRAYSCALE);
    if (sizeof img == 0) throw invalid_argument("cannot open image");
    return img;
}

template<typename _Tp> static  cv::Mat toMat(const vector<vector<_Tp> > vecIn) {
    cv::Mat_<_Tp> matOut(vecIn.size(), vecIn.at(0).size());
    for (int i = 0; i < matOut.rows; ++i) {
        for (int j = 0; j < matOut.cols; ++j) {
            matOut(i, j) = vecIn.at(i).at(j);
        }
    }
    return matOut;
}

Mat show_dft(Mat img, bool make_better) {
    Mat channels[2];
    Mat magn;
    split(img, channels);
    magnitude(channels[0], channels[1], magn);
    magn += Scalar::all(1);
    log(magn, magn);
    normalize(magn, magn, 0, 1, NormTypes::NORM_MINMAX);
    if(make_better)
        return make_spectr_better(magn);
    else return magn;
}


//Ядра фильтров
vector<vector<int8_t>> Sobel_x = {
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
};

vector<vector<int8_t>> Sobel_y = {
    {1,   2,  1},
    {0,   0,  0},
    {-1, -2, -1}
};

vector<vector<int8_t>> Laplasse = {
    {0,  1, 0},
    {1, -4, 1},
    {0,  1, 0}
};

vector<vector<int8_t>> Box = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 1}
};

void task_1(Mat img) {
    Mat convolute;
    //convolute = DFT_convolute(DFT_convolute(image, Sobel_y, convolute), Sobel_x, convolute);
    convolute = DFT_convolute(img, Laplasse, convolute);
    normalize(convolute, convolute, 0, 1, NORM_MINMAX);
    imshow("convolute", convolute);
}

void task_2(Mat img) {
    Mat spectr_og = return_img_spectr(img);
    imshow("OG image DFT", show_dft(spectr_og, true));
    Mat low_freq, high_freq;
    high_freq = return_high_frequence(spectr_og);
    low_freq = return_low_frequence(spectr_og);

    dft(high_freq, high_freq, DFT_INVERSE | DFT_REAL_OUTPUT);
    dft(low_freq, low_freq, DFT_INVERSE | DFT_REAL_OUTPUT);

    high_freq(Rect(0, 0, img.cols, img.rows)).copyTo(high_freq);
    low_freq(Rect(0, 0, img.cols, img.rows)).copyTo(low_freq);


    normalize(high_freq, high_freq, 0, 1, NORM_MINMAX);
    imshow("high_freq res", high_freq);
    normalize(low_freq, low_freq, 0, 1, NORM_MINMAX);
    imshow("low freq res", low_freq);
}

void task_3(Mat src_img, Mat template_img) {
    Mat result_mat;
    Mat Output = src_img.clone();
    Scalar srcMean = mean(src_img);
    Scalar templMean = mean(template_img);

    src_img = DFT_convolute(src_img, Laplasse, src_img);
    template_img = DFT_convolute(template_img, Laplasse, template_img);

    src_img.convertTo(src_img, CV_8U);
    template_img.convertTo(template_img, CV_8U);

    //Вот тут перед тем, как включить функцию корреляции - я вычитанию средние значения
    Mat src_mean(src_img.rows, src_img.cols, src_img.type(), Scalar::all(srcMean[0]));
    Mat template_mean(template_img.rows, template_img.cols, template_img.type(), Scalar::all(templMean[0]));

    imshow("src Mean", src_mean);
    imshow("template Mean", template_mean);
    
    src_img = src_img - src_mean;
    cv::normalize(src_img, src_img, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    imshow("src after subtract Mean", src_img);
    src_img.convertTo(src_img, CV_32FC1);

    
    template_img = template_img - template_mean;
    cv::normalize(template_img, template_img, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());
    imshow("template after subtract Mean", template_img);
    template_img.convertTo(template_img, CV_32FC1);

    result_mat = DFT_Correlation(src_img, template_img);
    cv::normalize(result_mat, result_mat, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    cv::imshow("Correlation", result_mat);

    double minVal; double maxVal;
    cv::Point minLoc, maxLoc, matchLoc;
    cv::minMaxLoc(result_mat, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
    matchLoc = maxLoc;
    threshold(result_mat, result_mat, maxVal - 0.01, 255, CV_8U);
    imshow("Threshold", result_mat);

    cv::rectangle(
        Output,
        matchLoc,
        cv::Point(matchLoc.x + template_img.cols, matchLoc.y + template_img.rows),
        CV_RGB(255, 0, 0),
        1);

    imshow("Output", Output);
}


int main(){
    image = open_image("250px-Fourier2.jpg");
    //imshow("image", image);
    image.convertTo(image, CV_32FC1);


    gos_znak = open_image("license_plate.png");
    imshow("Znak", gos_znak);
    gos_znak.convertTo(gos_znak, CV_32FC1);

 

    _symbol = open_image("5.jpg");
    imshow("Symbol", _symbol);
    _symbol.convertTo(_symbol, CV_32FC1);


    //task_1(image);
    
    //task_2(image);

    task_3(gos_znak, _symbol);

    waitKey();
}

Mat DFT_convolute(Mat img, vector<vector<int8_t>> filter, Mat Output){
    Size dftSize;
    Mat Filter;
    Filter = toMat(filter);
    Filter.convertTo(Filter, CV_32FC1);

    Output.create(img.rows, img.cols, img.type());

    //Получаем оптимальные размеры холста:
    dftSize.width = getOptimalDFTSize(img.cols + Filter.cols - 1);
    dftSize.height = getOptimalDFTSize(img.rows + Filter.rows - 1);
    
    //Масштабируем фильтр и исходное изображение до размеров холста
    Mat canvas_image(dftSize, img.type(), Scalar::all(0));
    Mat canvas_filter(dftSize, Filter.type(), Scalar::all(0));

    Mat insert_region_image(canvas_image, Rect(0, 0, img.cols, img.rows));
    Mat insert_region_filter(canvas_filter, Rect(0, 0, Filter.cols, Filter.rows));

    img.copyTo(insert_region_image);
    Filter.copyTo(insert_region_filter);

    //ДПФ
    dft(canvas_image, canvas_image, DFT_COMPLEX_OUTPUT);
    dft(canvas_filter, canvas_filter, DFT_COMPLEX_OUTPUT);

    //Вывести красиво образы Фурье исходного изображения и фильтра
    imshow("og image dft", show_dft(canvas_image, true));
    imshow("fiter dft", show_dft(canvas_filter, true));


    //Проводим свертку, поэтому в конце false
    mulSpectrums(canvas_image, canvas_filter, canvas_image, 0, false); 

    imshow("convolute dft", show_dft(canvas_image, false));

    //Обратное ДПФ - уже свернутого с фильтром изображения
    dft(canvas_image, canvas_image, DFT_INVERSE | DFT_REAL_OUTPUT);

    //Обрезаем холст до размеров исходного изображения
    
    canvas_image(Rect(0, 0, Output.cols, Output.rows)).copyTo(Output);

    //imshow("canvas", canvas_filter);
    return Output;
}

Mat DFT_Correlation(Mat img1, Mat img2){
    Size dftSize;
    Mat Output;
   // Output = img1.clone();
    //Output.create(img1.rows, img1.cols, img1.type());

    //Получаем оптимальные размеры холста:
    dftSize.width = getOptimalDFTSize(img1.cols + img2.cols - 1);
    dftSize.height = getOptimalDFTSize(img1.rows + img2.rows - 1);

    //Масштабируем фильтр и исходное изображение до размеров холста
    Mat canvas_image1(dftSize, img1.type(), Scalar::all(0));
    Mat canvas_image2(dftSize, img2.type(), Scalar::all(0));

    Mat insert_region_image1(canvas_image1, Rect(0, 0, img1.cols, img1.rows));
    Mat insert_region_image2(canvas_image2, Rect(0, 0, img2.cols, img2.rows));

    img1.copyTo(insert_region_image1);
    img2.copyTo(insert_region_image2);

    //ДПФ
    dft(canvas_image1, canvas_image1, DFT_COMPLEX_OUTPUT);
    dft(canvas_image2, canvas_image2, DFT_COMPLEX_OUTPUT);

    //Вывести красиво образы Фурье исходного изображения и фильтра
    imshow("znak dft", show_dft(canvas_image1, true));
    imshow("symbol dft", show_dft(canvas_image2, true));

    mulSpectrums(canvas_image1, canvas_image2, canvas_image1, 0, true);
    

    imshow("convolution dft", show_dft(canvas_image1, true));

    //Обратное ДПФ - уже свернутого с фильтром изображения
    dft(canvas_image1, canvas_image1, DFT_INVERSE | DFT_REAL_OUTPUT);

    canvas_image1.copyTo(Output);
    return Output;
}



Mat return_high_frequence(Mat src){
    Mat spectr = make_spectr_better(src);
    Mat mask(Size(spectr.cols, spectr.rows), CV_8U, Scalar::all(255));
    Point center(spectr.cols / 2, spectr.rows / 2);

    circle(mask, center, spectr.cols / 4, Scalar::all(0), -1);
    Mat output;
    spectr.copyTo(output, mask);
    imshow("high freq", show_dft(output, false));
    return output;
}

Mat return_low_frequence(Mat src){
    Mat spectr = make_spectr_better(src);
    Mat mask(Size(spectr.cols, spectr.rows), CV_8U, Scalar::all(0));
    Point center(spectr.cols / 2, spectr.rows / 2);

    circle(mask, center, spectr.cols / 4, Scalar::all(255), -1);
    Mat output;
    spectr.copyTo(output, mask);
    imshow("low freq", show_dft(output, false));
    return output;
}
