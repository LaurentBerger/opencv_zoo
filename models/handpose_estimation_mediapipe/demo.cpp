#include <vector>
#include <string>
#include <utility>
#include <cmath>
const long double _M_PI = 3.141592653589793238L;

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace dnn;

vector< pair<dnn::Backend, dnn::Target> > backendTargetPairs = {
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_OPENCV, dnn::DNN_TARGET_CPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CUDA, dnn::DNN_TARGET_CUDA_FP16),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_TIMVX, dnn::DNN_TARGET_NPU),
        std::make_pair<dnn::Backend, dnn::Target>(dnn::DNN_BACKEND_CANN, dnn::DNN_TARGET_NPU) };

class MPHandPose {
    Net net;
    string modelPath;
    float confThreshold;
    float nmsThreshold;
    Size inputSize;
    const vector<int> palmLandmarkIds{ 0, 5, 9, 13, 17, 1, 2 };
    const int palmLandmarksIndexofMiddleFingerBase = 2;
    const int palmLandmarksIndexofPalmBase = 0;
    Mat palmBoxPreShiftVector;
    const int palmBoxPreEnlargeFactor = 4;
    Mat palmBoxShiftVector;
    const int palmBoxEnlargeFactor = 3;
    Mat handBoxShiftVector;
    const float handBoxEnlargeFactor = 1.65f;
    dnn::Backend backendId;
    dnn::Target targetId;
public:
    MPHandPose(string modPath, float confThresh = 0.5, dnn::Backend bId = DNN_BACKEND_DEFAULT, dnn::Target tId = DNN_TARGET_CPU):
        modelPath(modPath), confThreshold(confThresh),
        backendId(bId), targetId(tId)
    {
        samples::addSamplesDataSearchPath("c:/lib/opencv_zoo/models/palm_detection_mediapipe/");
        samples::addSamplesDataSearchPath("c:/lib/opencv_zoo/models/handpose_estimation_mediapipe/");
        palmBoxPreShiftVector =  (Mat_<float>(1, 2) << 0, 0 );
        palmBoxShiftVector = (Mat_<float>(1, 2) << 0.f, -0.4f);
        handBoxShiftVector = (Mat_<float>(1, 2) << 0.f, -0.1f);
        this->inputSize = Size(224, 224);
        Point x(0, 0);
        this->net = dnn::readNet(samples::findFile(this->modelPath));
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
    }
    void setBackendAndTarget(dnn::Backend bId, dnn::Target tId)
    {
        this->backendId = bId;
        this->targetId = tId;
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
    }

    tuple<Mat, Mat, Mat> cropAndPadFromPalm(Mat image, Mat palmBbox, bool forRotation=false)
    {
    // shift bounding box
        Mat whPalmBbox = palmBbox.row(1) - palmBbox.row(0);
        Mat shiftVector;
        float enlargeScale;
        if (forRotation)
            shiftVector = this->palmBoxPreShiftVector;
        else
            shiftVector =this->palmBoxShiftVector;
        multiply(shiftVector, whPalmBbox, shiftVector);
        palmBbox.row(1) = palmBbox.row(1) + shiftVector;
        palmBbox.row(0) = palmBbox.row(0) + shiftVector;
        // enlarge bounding box
        Mat centerPalmBbox, handBbox;
        reduce(palmBbox, centerPalmBbox, 0, REDUCE_AVG, CV_32F);
        centerPalmBbox = centerPalmBbox;
        whPalmBbox = palmBbox.row(1) - palmBbox.row(0);
        if (forRotation)
            enlargeScale = this->palmBoxPreEnlargeFactor;
        else
            enlargeScale = this->palmBoxEnlargeFactor;

        Mat newHalfSize = whPalmBbox * enlargeScale / 2;
        vector<Mat> vmat(2);
        vmat[0] = centerPalmBbox - newHalfSize;
        vmat[1] = centerPalmBbox + newHalfSize;
        vconcat(vmat, handBbox);
        handBbox.convertTo(palmBbox, CV_32S);

        Mat idx = palmBbox.col(0) < 0;
        palmBbox.col(0).setTo(0, idx);
        idx = palmBbox.col(0) >= image.cols;
        palmBbox.col(0).setTo(image.cols, idx);
        idx = palmBbox.col(1) < 0;
        palmBbox.col(1).setTo(0, idx);
        idx = palmBbox.col(1) >= image.rows;
        palmBbox.col(1).setTo(image.rows, idx);        // crop to the size of interest

        image = image(Rect(palmBbox.at<int>(0, 0), palmBbox.at<int>(0, 1), palmBbox.at<int>(1, 0) - palmBbox.at<int>(0, 0), palmBbox.at<int>(1, 1) - palmBbox.at<int>(0, 1)));
        int sideLen;
        if (forRotation)
            sideLen = pow(image.rows * image.rows + image.cols * image.cols, 0.5);
        else
            sideLen = max(image.rows, image.cols);

        int padH = sideLen - image.rows;
        int padW = sideLen - image.cols;
        int left = padW / 2;
        int top = padH / 2;
        int right = padW - left;
        int bottom = padH - top;
        copyMakeBorder(image, image, top, bottom, left, right, BORDER_CONSTANT, Scalar::all(0));
        Mat bias = palmBbox.row(0) - (Mat_<int>(1,2) <<left, top);
        return tuple<Mat, Mat, Mat>(image, palmBbox, bias);
    }

    tuple<Mat, Mat, float, Mat, Mat> preprocess(Mat img, Mat palm)
    {
    /**
                Rotate input for inference.
                Parameters:
            image - input image of BGR channel order
                palm_bbox - palm bounding box found in image of format [[x1, y1], [x2, y2]] (top - left and bottom - right points)
                palm_landmarks - 7 landmarks(5 finger base points, 2 palm base points) of shape[7, 2]
                Returns :
                rotated_hand - rotated hand image for inference
                rotate_palm_bbox - palm box of interest range
                angle - rotate angle for hand
                rotation_matrix - matrix for rotation and de - rotation
                pad_bias - pad pixels of interest range
    */
            // crop and pad image to interest range
        Mat padBias=(Mat_<int>(1,2) << 0, 0 );
        Mat palmBbox = palm.row(0).colRange(0, 4).reshape(0, 2);
        tuple<Mat, Mat, Mat> x = this->cropAndPadFromPalm(img, palmBbox, true);
        Mat bias = get<2>(x);
        img = get<0>(x);
        cvtColor(img, img, COLOR_BGR2RGB);

        padBias += bias;
        get<1>(x).convertTo(palmBbox, CV_32F);
        // Rotate input to have vertically oriented hand image
        // compute rotation
        palmBbox.row(0) -= padBias;
        palmBbox.row(1) -= padBias;
        Mat palmLandmarks = palm.colRange(4, 18).reshape(0, 7);
        Mat padBiasF;
        padBias.convertTo(padBiasF, CV_32F);
        for (int i = 0; i < palmLandmarks.rows; i++)
            palmLandmarks.row(i) = palmLandmarks.row(i) - padBiasF;
        Mat p1 = palmLandmarks.row(this->palmLandmarksIndexofPalmBase);
        Mat p2 = palmLandmarks.row(this->palmLandmarksIndexofMiddleFingerBase);
        float radians = _M_PI / 2 - atan2(-(p2.at<float>(1) - p1.at<float>(1)), p2.at<float>(0) - p1.at<float>(0));
        radians = radians - 2 * _M_PI * int((radians + _M_PI) / (2 * _M_PI));
        float angle = radians * 180 / _M_PI;
        //  get bbox center
        Mat centerPalmBbox;
        reduce(palmBbox, centerPalmBbox, 0, REDUCE_AVG, CV_32F);
        //  get rotation matrix
        Mat rotationMatrix = getRotationMatrix2D(Point2f(centerPalmBbox.at<float>(0), centerPalmBbox.at<float>(1)), angle, 1.0);
        //  get rotated image
        Mat rotatedImage;
        warpAffine(img, rotatedImage, rotationMatrix, Size(img.cols, img.rows));
        // get bounding boxes from rotated palm landmarks
        Mat homogeneousCoord(3, palmLandmarks.rows, CV_64F, Scalar::all(1));
        Mat p=palmLandmarks.t();
        p.copyTo(homogeneousCoord.rowRange(0, 2));
        Mat rotatedPalmLandmarks = rotationMatrix * homogeneousCoord;
            
        //  get landmark bounding box
        int aMin, aMax;
        double vMin, vMax;
        Mat rotatedPalmBbox(2, 2, CV_32F);
        for (int i = 0; i <rotatedPalmBbox.rows; i++)
        {
            minMaxIdx(rotatedPalmLandmarks.row(i), &vMin, &vMax, nullptr, nullptr);
            rotatedPalmBbox.at<float>(0, i) = vMin;
            rotatedPalmBbox.at<float>(1, i) = vMax;
        }
        tuple<Mat, Mat, Mat> cropTuple = this->cropAndPadFromPalm(rotatedImage, rotatedPalmBbox);
        Mat blob1, blob2;
        resize(get<0>(cropTuple), blob1, this->inputSize, INTER_AREA);
        blob1.convertTo(blob2, CV_32FC3, 1 / 255.);
        int sz[] = { 1,blob2.rows, blob2.cols, blob2.channels() };
        Mat blobF(4, sz, CV_32F); 
        blob2.copyTo(Mat(blob2.rows, blob2.cols, CV_32FC3, blobF.ptr(0)));
        return tuple<Mat, Mat, float, Mat, Mat>(blobF, get<1>(cropTuple), angle, rotationMatrix, padBias);
    }

    Mat infer(Mat srcImg, Mat palm)
    {
        tuple<Mat, Mat, float, Mat, Mat> x = this->preprocess(srcImg, palm);
        Mat inputBlob = get<0>(x);
        Mat rotatedPalmBbox = get<1>(x);
        float angle = get<2>(x);;
        Mat rotationMatrix= get<3>(x);
        Mat padBias = get<4>(x);;
        // Forward
        this->net.setInput(inputBlob);


        vector<Mat> outputBlobs;
        this->net.forward(outputBlobs, this->net.getUnconnectedOutLayersNames());

        // Postprocess
        Mat results = this->postprocess(outputBlobs, rotatedPalmBbox, angle, rotationMatrix, padBias);
        return results; // [bbox_coords, landmarks_coords, conf]
    }

    Mat postprocess(vector<Mat> outputs, Mat rotatedPalmBbox, float angle, Mat rotationMatrix, Mat padBias)
    {
        Mat landMarks = outputs[0];
        Mat conf = outputs[1];
        Mat handedness = outputs[2];
        Mat landMarksWord = outputs[3];

        if (conf.at<float>(0) < this->confThreshold)
            return Mat();
        landMarks = landMarks.row(0).reshape(0, 21);  // shape: (1, 63) -> (21, 3)
        landMarksWord = landMarksWord.row(0).reshape(0, 21); // shape : (1, 63) -> (21, 3)
        //# transform coords back to the input coords
        Mat whRrotatedPalmBbox = rotatedPalmBbox.row(1) - rotatedPalmBbox.row(0);
        Size2f scaleFactor(whRrotatedPalmBbox.at<int>(0) / float(this->inputSize.width), whRrotatedPalmBbox.at<int>(1) / float(this->inputSize.height));
        for (int i = 0; i < landMarks.rows; i++)
        {
            landMarks.at<float>(i, 0) = (landMarks.at<float>(i, 0) - this->inputSize.width / 2) * max(scaleFactor.width, scaleFactor.height);
            landMarks.at<float>(i, 1) = (landMarks.at<float>(i, 1) - this->inputSize.height / 2) * max(scaleFactor.width, scaleFactor.height);
            landMarks.at<float>(i, 2) = landMarks.at<float>(i, 2) * max(scaleFactor.width, scaleFactor.height); // depth scaling
        }
        Mat coordsRotationMatrix = getRotationMatrix2D(Point(0, 0), angle, 1.0);
        coordsRotationMatrix.convertTo(coordsRotationMatrix, CV_32F);
        Mat rotatedLandmarks = (coordsRotationMatrix.colRange(0, 2) * landMarks.colRange(0, 2).t()).t();
        hconcat(rotatedLandmarks, landMarks.col(2), rotatedLandmarks);
        Mat rotatedLandmarksWord = (coordsRotationMatrix.colRange(0, 2) * landMarksWord.colRange(0, 2).t()).t();
        hconcat(rotatedLandmarksWord, landMarksWord.col(2), rotatedLandmarksWord);
        // invert rotation
        Mat rotationComponent = rotationMatrix.colRange(0, 2).t();
        Mat translationComponent = rotationMatrix.col(2);
        Mat invertedTranslation = -rotationComponent * translationComponent;
        Mat inverseRotationMatrix;
        hconcat(rotationComponent, invertedTranslation, inverseRotationMatrix);
        // get box center
        Mat center;
        rotatedPalmBbox.convertTo(rotatedPalmBbox, CV_32F);
        reduce(rotatedPalmBbox, center, 0, REDUCE_AVG, CV_64F);
        hconcat(center, Mat::ones(1,1,CV_64FC1), center);
        Mat originalCenter = inverseRotationMatrix * center.t();
        for (int i = 0; i < landMarks.rows; i++)
        {
            landMarks.at<float>(i, 0) = rotatedLandmarks.at<float>(i, 0) + originalCenter.at<double>(0) + padBias.at<int>(0);
            landMarks.at<float>(i, 1) = rotatedLandmarks.at<float>(i, 1) + originalCenter.at<double>(1) + padBias.at<int>(1);
        }
        // get bounding box from rotated_landmarks
        double top, left, right, bottom;
        minMaxLoc(landMarks.col(0), &left, &right);
        minMaxLoc(landMarks.col(1), &top, &bottom);
        Mat bbox = (Mat_<float>(2, 2) << top, left, bottom, right);
        // shift bounding box
        Mat whBox = bbox.row(1) - bbox.row(0);
        Mat shiftVector= (Mat_<float>(1,2) << this->handBoxShiftVector.at<float>(0) * whBox.at<float>(0), this->handBoxShiftVector.at<float>(1)* whBox.at<float>(1));
        bbox.row(0) = bbox.row(0) + shiftVector;
        bbox.row(1) = bbox.row(1) + shiftVector;
        // enlarge bounding box
        Mat centerBox;
        reduce(bbox, centerBox, 0, REDUCE_AVG);
        bbox.row(0) = bbox.row(0) + shiftVector;
        whBox = bbox.row(1) - bbox.row(0);
        Mat newHalfSize= whBox * this->handBoxEnlargeFactor / 2;
        vector<Mat> vmat(2 + landMarks.rows + rotatedLandmarksWord.rows + 2 );
        vmat[0] = centerBox - newHalfSize;
        vmat[1] = centerBox + newHalfSize;
        for (int i = 0; i < landMarks.rows; i++)
        {
            vmat[2 + i] = landMarks.row(i);
        }
        for (int i = 0; i < rotatedLandmarksWord.rows; i++)
        {
            vmat[2 + landMarks.rows + i] = rotatedLandmarksWord.row(i);
        }
        vmat[2 + landMarks.rows + rotatedLandmarksWord.rows] = handedness;
        vmat[2 + landMarks.rows + rotatedLandmarksWord.rows+1] = conf;
        hconcat(vmat, bbox);

        // [0: 4]: hand bounding box found in image of format [x1, y1, x2, y2] (top-left and bottom-right points)
        // [4: 67]: screen landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], z value is relative to WRIST
        // [67: 130]: world landmarks with format [x1, y1, z1, x2, y2 ... x21, y21, z21], 3D metric x, y, z coordinate
        // [130]: handedness, (left)[0, 1](right) hand
        // [131]: confidence
        return bbox;
    }

};

class MPPalmDet {
private:
    Net net;
    string modelPath;
    Size inputSize;
    Size originalSize;
    Image2BlobParams paramMediapipe;
    float scoreThreshold;
    float nmsThreshold;
    float topK;
    dnn::Backend backendId;
    dnn::Target targetId;
    Mat anchors;

public:
    MPPalmDet(string modPath, float nmsThresh=0.5, float scoreThresh=0.3, float topkVal = 5000, dnn::Backend bId=DNN_BACKEND_DEFAULT, dnn::Target tId=DNN_TARGET_CPU) :
        modelPath(modPath), scoreThreshold(scoreThresh),
        nmsThreshold(nmsThresh), topK(topkVal),
        backendId(bId), targetId(tId)
    {
        samples::addSamplesDataSearchPath("c:/lib/opencv_zoo/models/palm_detection_mediapipe/");
        this->net = readNet(samples::findFile(this->modelPath));
        this->inputSize = Size(192, 192);
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
        this->generateAnchors();
    }

    void setBackendAndTarget(dnn::Backend bId, dnn::Target tId)
    {
        this->backendId = bId;
        this->targetId = tId;
        this->net.setPreferableBackend(this->backendId);
        this->net.setPreferableTarget(this->targetId);
    }

    Mat preprocess(Mat img)
    {
        Mat blob;
       
        this->paramMediapipe.datalayout = DNN_LAYOUT_NHWC;
        this->paramMediapipe.paddingmode = DNN_PMODE_LETTERBOX;
        this->paramMediapipe.ddepth = CV_32F;
        this->paramMediapipe.mean = Scalar::all(0);
        this->paramMediapipe.scalefactor = Scalar::all(1/255.0);
        this->paramMediapipe.size = this->inputSize;
        this->paramMediapipe.swapRB = true;
        this->originalSize = img.size();
        blob = blobFromImageWithParams(img, paramMediapipe);
        return blob;
    }

   Mat infer(Mat srcimg)
   {
        Mat inputBlob = this->preprocess(srcimg);

        this->net.setInput(inputBlob);
        vector<Mat> outs;
        this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

        Mat predictions = this->postprocess(outs);
        return predictions;
   }

   Mat postprocess(vector<Mat> outputs)
   {
        Mat scores;
        Mat boxAndlandmarkDelta;
        Mat boxDelta;
        Mat landMarkDelta;
        double scale = max(originalSize.width, originalSize.height);
        scores = outputs[1].reshape(0, outputs[1].size[0]);

        boxAndlandmarkDelta = outputs[0].reshape(outputs[0].size[0], outputs[0].size[1]); 
        boxDelta = boxAndlandmarkDelta.colRange(0, 4);
        landMarkDelta = boxAndlandmarkDelta.colRange(4, boxAndlandmarkDelta.cols);
        // get scores 
        exp(-scores, scores);
        Mat deno = 1 + scores; ;
        divide(1.0, deno, scores);
        // get boxes
        vector<Rect> rBox(boxDelta.rows), rImg;
        Mat tl, dimRect;
 
        for (int row = 0; row < boxDelta.rows; row++)
        {
            Rect r(boxDelta.at<float>(row, 0) - boxDelta.at<float>(row, 2) / 2 + this->anchors.at<float>(row, 0) * this->inputSize.width, 
                   boxDelta.at<float>(row, 1) - boxDelta.at<float>(row, 3) / 2 + this->anchors.at<float>(row, 1) * this->inputSize.height,
                   boxDelta.at<float>(row, 2),
                   boxDelta.at<float>(row, 3));
            rBox[row] = r;
        }   
        blobRectsToImageRects(rBox, rImg, this->originalSize, this->paramMediapipe);
        vector< int > keep;
        NMSBoxes(rBox, scores, this->scoreThreshold, this->nmsThreshold, keep, 1.0f, this->topK);
        if (keep.size() == 0)
            return Mat();
        Mat result(keep.size(), boxDelta.cols + landMarkDelta.cols + 1, CV_32FC1, Scalar::all(0));
        int row = 0;
        for (auto idx : keep)
        {
            result.at<float>(row, 0) = rImg[idx].tl().x;
            result.at<float>(row, 1) = rImg[idx].tl().y;
            result.at<float>(row, 2) = rImg[idx].br().x;
            result.at<float>(row, 3) = rImg[idx].br().y;
            for (int i = 0; i < landMarkDelta.cols / 2; i++)
            {
                Rect r(landMarkDelta.at<float>(idx, 2 * i) + this->anchors.at<float>(idx, 0) * this->inputSize.width,
                    landMarkDelta.at<float>(idx, 2 * i + 1) + this->anchors.at<float>(idx, 1) * this->inputSize.height,
                    1, 1);
                r = blobRectToImageRect(r, this->originalSize, this->paramMediapipe);
                result.at<float>(row, 4 + 2 * i) = r.x;
                result.at<float>(row, 4 + 2 * i + 1) = r.y;
            }
            result.at<float>(row, boxDelta.cols + landMarkDelta.cols ) = scores.at<float>(idx);
            row++;
        }
        return result;
   }


    void generateAnchors();
};



class GestureClassification
{
protected:
    double vector2Angle(Mat v1, Mat v2)
    {

        Mat uv1 = v1 / norm(v1);
        Mat uv2 = v2 / norm(v2);
        double angle = acos(uv1.dot(uv2)) / _M_PI * 180;
        return angle;
    }
    vector<double> handAngle(Mat hand)
    {
        vector<double> angleList;
        // thumb
        angleList.push_back(vector2Angle(hand.row(0) - hand.row(2), hand.row(3) - hand.row(4)));
        // index
        angleList.push_back(vector2Angle(hand.row(0) - hand.row(6), hand.row(7) - hand.row(8)));
        // middle
        angleList.push_back(vector2Angle(hand.row(0) - hand.row(10), hand.row(11) - hand.row(12)));
        // ring
        angleList.push_back(vector2Angle(hand.row(0) - hand.row(14), hand.row(15) - hand.row(16)));
        // pink
        angleList.push_back(vector2Angle(hand.row(0) - hand.row(18), hand.row(197) - hand.row(20)));
        return angleList;
    }
    vector<bool> fingerStatus(Mat lmList)
    {
        vector<bool> fingerList;

        Point origin = lmList.row(0);
        pair<int, int> keypointList[] = { make_pair(5, 4),make_pair(6, 8),make_pair(10, 12),make_pair(14, 16), make_pair(18, 20) };
        for (auto ref : keypointList)
        {
            Point p1 = lmList.row(ref.first);
            Point p2 = lmList.row(ref.second);
            if (norm(p2 - origin) > norm(p1 - origin))
                fingerList.push_back(true);
            else
                fingerList.push_back(false);
        }
        return fingerList;
    }

    string classifyhand(Mat  hand)
    {
        double thrAngle = 65.;
        double thrAngleThumb = 30.;
        double thrAngleS = 49.;
        string gestureStr = "Undefined";

        vector<double> angleList = handAngle(hand);

        vector<bool> fgSt = fingerStatus(hand);
        bool thumbOpen = fgSt[0];
        bool firstOpen = fgSt[1];
        bool secondOpen = fgSt[2];
        bool thirdOpen = fgSt[3];
        bool fourthOpen = fgSt[4];
        // Number
        if ((angleList[0] > thrAngleThumb) && (angleList[1] > thrAngle) && (angleList[2] > thrAngle) && (
            angleList[3] > thrAngle) && (angleList[4] > thrAngle) &&
            !firstOpen && !secondOpen && !thirdOpen && !fourthOpen)
            gestureStr = "Zero";
        else if ((angleList[0] > thrAngleThumb) && (angleList[1] < thrAngleS) && (angleList[2] > thrAngle) && (
            angleList[3] > thrAngle) && (angleList[4] > thrAngle) && \
            firstOpen && !secondOpen && !thirdOpen && !fourthOpen)
            gestureStr = "One";
        else if ((angleList[0] > thrAngleThumb) && (angleList[1] < thrAngleS) && (angleList[2] < thrAngleS) && (
            angleList[3] > thrAngle) && (angleList[4] > thrAngle) && \
            !thumbOpen && firstOpen && secondOpen && !thirdOpen && !fourthOpen)
            gestureStr = "Two";
        else if ((angleList[0] > thrAngleThumb) && (angleList[1] < thrAngleS) && (angleList[2] < thrAngleS) && (
            angleList[3] < thrAngleS) && (angleList[4] > thrAngle) && \
            !thumbOpen && firstOpen && secondOpen && thirdOpen && !fourthOpen)
            gestureStr = "Three";
        else if ((angleList[0] > thrAngleThumb) && (angleList[1] < thrAngleS) && (angleList[2] < thrAngleS) && (
                angleList[3] < thrAngleS) && (angleList[4] < thrAngle) && \
            firstOpen && secondOpen && thirdOpen && fourthOpen)
            gestureStr = "Four";
        else if ((angleList[0] < thrAngleS) && (angleList[1] < thrAngleS) && (angleList[2] < thrAngleS) && (
                angleList[3] < thrAngleS) && (angleList[4] < thrAngleS) && \
            thumbOpen && firstOpen && secondOpen && thirdOpen && fourthOpen)
            gestureStr = "Five";
        else if ((angleList[0] < thrAngleS) && (angleList[1] > thrAngle) && (angleList[2] > thrAngle) && (
                angleList[3] > thrAngle) && (angleList[4] < thrAngleS) && \
            thumbOpen &&  !firstOpen &&  !secondOpen &&  !thirdOpen && fourthOpen)
            gestureStr = "Six";
        else if ((angleList[0] < thrAngleS) && (angleList[1] < thrAngle) && (angleList[2] > thrAngle) && (
                angleList[3] > thrAngle) && (angleList[4] > thrAngleS) && \
            thumbOpen && firstOpen &&  !secondOpen &&  !thirdOpen &&  !fourthOpen)
            gestureStr = "Seven";
        else if ((angleList[0] < thrAngleS) && (angleList[1] < thrAngle) && (angleList[2] < thrAngle) && (
                angleList[3] > thrAngle) && (angleList[4] > thrAngleS) && \
            thumbOpen && firstOpen && secondOpen &&  !thirdOpen &&  !fourthOpen)
            gestureStr = "Eight";
        else if ((angleList[0] < thrAngleS) && (angleList[1] < thrAngle) && (angleList[2] < thrAngle) && (
                angleList[3] < thrAngle) && (angleList[4] > thrAngleS) && \
            thumbOpen && firstOpen && secondOpen && thirdOpen &&  !fourthOpen)
            gestureStr = "Nine";

        return gestureStr;
    }
public:
    string classify(Mat landmarks)
    {
        Mat hand = landmarks.rowRange(0, 21).colRange(0, 2);
        string gesture = this->classifyhand(hand);
        return gesture;
    }
};

std::string keys =
"{ help  h          |                                               | Print help message. }"
"{ model m          | handpose_estimation_mediapipe_2023feb.onnx    | Usage: Path to the model, defaults to handpose_estimation_mediapipe_2023feb.onnx  }"
"{ input i          |                                               | Path to input image or video file. Skip this argument to capture frames from a camera.}"
"{ confidence       | 0.9                                           | Class confidence }"
"{ save s           | true                                          | Specify to save results. This flag is invalid when using camera. }"
"{ vis v            | 1                                             | Specify to open a window for result visualization. This flag is invalid when using camera. }"
"{ backend bt       | 0                                             | Choose one of computation backends: "
"0: (default) OpenCV implementation + CPU, "
"1: CUDA + GPU (CUDA), "
"2: CUDA + GPU (CUDA FP16), "
"3: TIM-VX + NPU, "
"4: CANN + NPU}";

void drawLines(Mat image, Mat landmarks, Mat keeplandmarks, bool isDrawPoint = true, int thickness = 2)
{

    vector<pair<int, int>> segment = {
        make_pair(0, 1), make_pair(1, 2), make_pair(2, 3), make_pair(3, 7),
        make_pair(0, 4), make_pair(4, 5), make_pair(5, 6), make_pair(6, 8),
        make_pair(9, 10),
        make_pair(12, 14), make_pair(14, 16), make_pair(16, 22), make_pair(16, 18), make_pair(16, 20), make_pair(18, 20),
        make_pair(11, 13), make_pair(13, 15), make_pair(15, 21), make_pair(15, 19), make_pair(15, 17), make_pair(17, 19),
        make_pair(11, 12), make_pair(11, 23), make_pair(23, 24), make_pair(24, 12),
        make_pair(24, 26), make_pair(26, 28), make_pair(28, 30), make_pair(28, 32), make_pair(30, 32),
        make_pair(23, 25), make_pair(25, 27),make_pair(27, 31), make_pair(27, 29), make_pair(29, 31) };
    for (auto p : segment)
        if (keeplandmarks.at<uchar>(p.first) && keeplandmarks.at<uchar>(p.second))
            line(image, Point(landmarks.row(p.first)), Point(landmarks.row(p.second)), Scalar(255, 255, 255), thickness);
    if (isDrawPoint)
        for (int idxRow = 0; idxRow < landmarks.rows; idxRow++)
            if (keeplandmarks.at<uchar>(idxRow))
                circle(image, Point(landmarks.row(idxRow)), thickness, Scalar(0, 0, 255), -1);
}



pair<Mat, Mat> visualize(Mat image, Mat handsPose, float fps = -1)
{
    Mat displayScreen = image.clone();
    Mat display3d(400, 400, CV_8UC3, Scalar::all(0));
    line(display3d, Point(200, 0), Point(200, 400), Scalar(255, 255, 255), 2);
    line(display3d, Point(0, 200), Point(400, 200), Scalar(255, 255, 255), 2);
    putText(display3d, "Main View", Point(0, 12), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Top View", Point(200, 12), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Left View", Point(0, 212), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    putText(display3d, "Right View", Point(200, 212), FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 0, 255));
    bool isDraw = false;  // ensure only one person is drawn
    GestureClassification gc;

    for (int idxHand = 0; idxHand < handsPose.rows; idxHand++)
    {
        Mat hand = handsPose.row(idxHand);
        float conf = hand.at<float>(hand.cols - 1);
        float handedness = hand.at<float>(hand.cols - 2);
        string handednessText;
        if (handedness <= 0.5)
            handednessText = "Left";
        else
            handednessText = "Right";
        Mat bbox;
        hand.colRange(0, 4).convertTo(hand, CV_32S);
        Mat landmarksScreen;
        hand.colRange(4, 67).convertTo(landmarksScreen, CV_32S);
        landmarksScreen.reshape(0, 3);
        Mat landmarksPose;
        hand.colRange(67, 130).convertTo(landmarksPose, CV_32S);
        landmarksPose.reshape(0, 3);
        string gesture = gc.classify(landmarksScreen);
    }
    return pair<Mat, Mat>(displayScreen, display3d);
}


int main(int argc, char** argv)
{
    CommandLineParser parser(argc, argv, keys);

    parser.about("Use this script to run Yolox deep learning networks in opencv_zoo using OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    string model = parser.get<String>("model");
    float confThreshold = parser.get<float>("confidence");
    bool vis = parser.get<bool>("vis");
    bool save = parser.get<bool>("save");
    int backendTargetid = parser.get<int>("backend");

    if (model.empty())
    {
        CV_Error(Error::StsError, "Model file " + model + " not found");
    }
    // palm detector
    MPPalmDet palmDetector("../palm_detection_mediapipe/palm_detection_mediapipe_2023feb.onnx", 0.5, 0.3, 3000,
        backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);
    // handpose detector
    MPHandPose handposeDetector(model, confThreshold, backendTargetPairs[backendTargetid].first, backendTargetPairs[backendTargetid].second);

    //! [Open a video file or an image file or a camera stream]
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(samples::findFile(parser.get<String>("input")));
    else
        cap.open(0);
    if (!cap.isOpened())
        CV_Error(Error::StsError, "Cannot opend video or file");
    Mat frame, inputBlob;


    static const std::string kWinName = model;
    int nbInference = 0;
    TickMeter tm;
    while (waitKey(1) < 0)
    {
        cap >> frame;
        if (frame.empty())
        {
            cout << "Frame is empty" << endl;
            waitKey();
            break;
        }


        
        // Palm detector inference
        Mat    palms = palmDetector.infer(frame);
        Mat hands = Mat();
        // Estimate the pose of each hand
        tm.start();
        for (int i = 0; i < palms.rows; i++)
        {
            // Handpose detector inference
            Mat handpose = handposeDetector.infer(frame, palms.row(i));
            if (!handpose.empty())
            {
                if (hands.empty())
                    hands = handpose;
                else
                    vconcat(hands, handpose, hands);
            }
        }
        tm.stop();
        // Draw results on the input image
        pair<Mat, Mat> duoimg = visualize(frame, hands);
        if (palms.rows == 0)
            cout << "No palm detected!";
        else
        {
            cout << "Palm detected!";
            putText(duoimg.first, format("FPS: %5lf", tm.getFPS()), Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        }
        imshow("MediaPipe Handpose Detection Demo", duoimg.first);
        imshow("3D HandPose Demo", duoimg.second);
        tm.reset();
    }
    return 0;
}



void MPPalmDet::generateAnchors()
{
    this->anchors=(Mat_<float>(2016,2) << 0.02083333, 0.02083333,
        0.02083333, 0.02083333,
        0.0625, 0.02083333,
        0.0625, 0.02083333,
        0.10416666, 0.02083333,
        0.10416666, 0.02083333,
        0.14583333, 0.02083333,
        0.14583333, 0.02083333,
        0.1875, 0.02083333,
        0.1875, 0.02083333,
        0.22916667, 0.02083333,
        0.22916667, 0.02083333,
        0.27083334, 0.02083333,
        0.27083334, 0.02083333,
        0.3125, 0.02083333,
        0.3125, 0.02083333,
        0.35416666, 0.02083333,
        0.35416666, 0.02083333,
        0.39583334, 0.02083333,
        0.39583334, 0.02083333,
        0.4375, 0.02083333,
        0.4375, 0.02083333,
        0.47916666, 0.02083333,
        0.47916666, 0.02083333,
        0.5208333, 0.02083333,
        0.5208333, 0.02083333,
        0.5625, 0.02083333,
        0.5625, 0.02083333,
        0.6041667, 0.02083333,
        0.6041667, 0.02083333,
        0.6458333, 0.02083333,
        0.6458333, 0.02083333,
        0.6875, 0.02083333,
        0.6875, 0.02083333,
        0.7291667, 0.02083333,
        0.7291667, 0.02083333,
        0.7708333, 0.02083333,
        0.7708333, 0.02083333,
        0.8125, 0.02083333,
        0.8125, 0.02083333,
        0.8541667, 0.02083333,
        0.8541667, 0.02083333,
        0.8958333, 0.02083333,
        0.8958333, 0.02083333,
        0.9375, 0.02083333,
        0.9375, 0.02083333,
        0.9791667, 0.02083333,
        0.9791667, 0.02083333,
        0.02083333, 0.0625,
        0.02083333, 0.0625,
        0.0625, 0.0625,
        0.0625, 0.0625,
        0.10416666, 0.0625,
        0.10416666, 0.0625,
        0.14583333, 0.0625,
        0.14583333, 0.0625,
        0.1875, 0.0625,
        0.1875, 0.0625,
        0.22916667, 0.0625,
        0.22916667, 0.0625,
        0.27083334, 0.0625,
        0.27083334, 0.0625,
        0.3125, 0.0625,
        0.3125, 0.0625,
        0.35416666, 0.0625,
        0.35416666, 0.0625,
        0.39583334, 0.0625,
        0.39583334, 0.0625,
        0.4375, 0.0625,
        0.4375, 0.0625,
        0.47916666, 0.0625,
        0.47916666, 0.0625,
        0.5208333, 0.0625,
        0.5208333, 0.0625,
        0.5625, 0.0625,
        0.5625, 0.0625,
        0.6041667, 0.0625,
        0.6041667, 0.0625,
        0.6458333, 0.0625,
        0.6458333, 0.0625,
        0.6875, 0.0625,
        0.6875, 0.0625,
        0.7291667, 0.0625,
        0.7291667, 0.0625,
        0.7708333, 0.0625,
        0.7708333, 0.0625,
        0.8125, 0.0625,
        0.8125, 0.0625,
        0.8541667, 0.0625,
        0.8541667, 0.0625,
        0.8958333, 0.0625,
        0.8958333, 0.0625,
        0.9375, 0.0625,
        0.9375, 0.0625,
        0.9791667, 0.0625,
        0.9791667, 0.0625,
        0.02083333, 0.10416666,
        0.02083333, 0.10416666,
        0.0625, 0.10416666,
        0.0625, 0.10416666,
        0.10416666, 0.10416666,
        0.10416666, 0.10416666,
        0.14583333, 0.10416666,
        0.14583333, 0.10416666,
        0.1875, 0.10416666,
        0.1875, 0.10416666,
        0.22916667, 0.10416666,
        0.22916667, 0.10416666,
        0.27083334, 0.10416666,
        0.27083334, 0.10416666,
        0.3125, 0.10416666,
        0.3125, 0.10416666,
        0.35416666, 0.10416666,
        0.35416666, 0.10416666,
        0.39583334, 0.10416666,
        0.39583334, 0.10416666,
        0.4375, 0.10416666,
        0.4375, 0.10416666,
        0.47916666, 0.10416666,
        0.47916666, 0.10416666,
        0.5208333, 0.10416666,
        0.5208333, 0.10416666,
        0.5625, 0.10416666,
        0.5625, 0.10416666,
        0.6041667, 0.10416666,
        0.6041667, 0.10416666,
        0.6458333, 0.10416666,
        0.6458333, 0.10416666,
        0.6875, 0.10416666,
        0.6875, 0.10416666,
        0.7291667, 0.10416666,
        0.7291667, 0.10416666,
        0.7708333, 0.10416666,
        0.7708333, 0.10416666,
        0.8125, 0.10416666,
        0.8125, 0.10416666,
        0.8541667, 0.10416666,
        0.8541667, 0.10416666,
        0.8958333, 0.10416666,
        0.8958333, 0.10416666,
        0.9375, 0.10416666,
        0.9375, 0.10416666,
        0.9791667, 0.10416666,
        0.9791667, 0.10416666,
        0.02083333, 0.14583333,
        0.02083333, 0.14583333,
        0.0625, 0.14583333,
        0.0625, 0.14583333,
        0.10416666, 0.14583333,
        0.10416666, 0.14583333,
        0.14583333, 0.14583333,
        0.14583333, 0.14583333,
        0.1875, 0.14583333,
        0.1875, 0.14583333,
        0.22916667, 0.14583333,
        0.22916667, 0.14583333,
        0.27083334, 0.14583333,
        0.27083334, 0.14583333,
        0.3125, 0.14583333,
        0.3125, 0.14583333,
        0.35416666, 0.14583333,
        0.35416666, 0.14583333,
        0.39583334, 0.14583333,
        0.39583334, 0.14583333,
        0.4375, 0.14583333,
        0.4375, 0.14583333,
        0.47916666, 0.14583333,
        0.47916666, 0.14583333,
        0.5208333, 0.14583333,
        0.5208333, 0.14583333,
        0.5625, 0.14583333,
        0.5625, 0.14583333,
        0.6041667, 0.14583333,
        0.6041667, 0.14583333,
        0.6458333, 0.14583333,
        0.6458333, 0.14583333,
        0.6875, 0.14583333,
        0.6875, 0.14583333,
        0.7291667, 0.14583333,
        0.7291667, 0.14583333,
        0.7708333, 0.14583333,
        0.7708333, 0.14583333,
        0.8125, 0.14583333,
        0.8125, 0.14583333,
        0.8541667, 0.14583333,
        0.8541667, 0.14583333,
        0.8958333, 0.14583333,
        0.8958333, 0.14583333,
        0.9375, 0.14583333,
        0.9375, 0.14583333,
        0.9791667, 0.14583333,
        0.9791667, 0.14583333,
        0.02083333, 0.1875,
        0.02083333, 0.1875,
        0.0625, 0.1875,
        0.0625, 0.1875,
        0.10416666, 0.1875,
        0.10416666, 0.1875,
        0.14583333, 0.1875,
        0.14583333, 0.1875,
        0.1875, 0.1875,
        0.1875, 0.1875,
        0.22916667, 0.1875,
        0.22916667, 0.1875,
        0.27083334, 0.1875,
        0.27083334, 0.1875,
        0.3125, 0.1875,
        0.3125, 0.1875,
        0.35416666, 0.1875,
        0.35416666, 0.1875,
        0.39583334, 0.1875,
        0.39583334, 0.1875,
        0.4375, 0.1875,
        0.4375, 0.1875,
        0.47916666, 0.1875,
        0.47916666, 0.1875,
        0.5208333, 0.1875,
        0.5208333, 0.1875,
        0.5625, 0.1875,
        0.5625, 0.1875,
        0.6041667, 0.1875,
        0.6041667, 0.1875,
        0.6458333, 0.1875,
        0.6458333, 0.1875,
        0.6875, 0.1875,
        0.6875, 0.1875,
        0.7291667, 0.1875,
        0.7291667, 0.1875,
        0.7708333, 0.1875,
        0.7708333, 0.1875,
        0.8125, 0.1875,
        0.8125, 0.1875,
        0.8541667, 0.1875,
        0.8541667, 0.1875,
        0.8958333, 0.1875,
        0.8958333, 0.1875,
        0.9375, 0.1875,
        0.9375, 0.1875,
        0.9791667, 0.1875,
        0.9791667, 0.1875,
        0.02083333, 0.22916667,
        0.02083333, 0.22916667,
        0.0625, 0.22916667,
        0.0625, 0.22916667,
        0.10416666, 0.22916667,
        0.10416666, 0.22916667,
        0.14583333, 0.22916667,
        0.14583333, 0.22916667,
        0.1875, 0.22916667,
        0.1875, 0.22916667,
        0.22916667, 0.22916667,
        0.22916667, 0.22916667,
        0.27083334, 0.22916667,
        0.27083334, 0.22916667,
        0.3125, 0.22916667,
        0.3125, 0.22916667,
        0.35416666, 0.22916667,
        0.35416666, 0.22916667,
        0.39583334, 0.22916667,
        0.39583334, 0.22916667,
        0.4375, 0.22916667,
        0.4375, 0.22916667,
        0.47916666, 0.22916667,
        0.47916666, 0.22916667,
        0.5208333, 0.22916667,
        0.5208333, 0.22916667,
        0.5625, 0.22916667,
        0.5625, 0.22916667,
        0.6041667, 0.22916667,
        0.6041667, 0.22916667,
        0.6458333, 0.22916667,
        0.6458333, 0.22916667,
        0.6875, 0.22916667,
        0.6875, 0.22916667,
        0.7291667, 0.22916667,
        0.7291667, 0.22916667,
        0.7708333, 0.22916667,
        0.7708333, 0.22916667,
        0.8125, 0.22916667,
        0.8125, 0.22916667,
        0.8541667, 0.22916667,
        0.8541667, 0.22916667,
        0.8958333, 0.22916667,
        0.8958333, 0.22916667,
        0.9375, 0.22916667,
        0.9375, 0.22916667,
        0.9791667, 0.22916667,
        0.9791667, 0.22916667,
        0.02083333, 0.27083334,
        0.02083333, 0.27083334,
        0.0625, 0.27083334,
        0.0625, 0.27083334,
        0.10416666, 0.27083334,
        0.10416666, 0.27083334,
        0.14583333, 0.27083334,
        0.14583333, 0.27083334,
        0.1875, 0.27083334,
        0.1875, 0.27083334,
        0.22916667, 0.27083334,
        0.22916667, 0.27083334,
        0.27083334, 0.27083334,
        0.27083334, 0.27083334,
        0.3125, 0.27083334,
        0.3125, 0.27083334,
        0.35416666, 0.27083334,
        0.35416666, 0.27083334,
        0.39583334, 0.27083334,
        0.39583334, 0.27083334,
        0.4375, 0.27083334,
        0.4375, 0.27083334,
        0.47916666, 0.27083334,
        0.47916666, 0.27083334,
        0.5208333, 0.27083334,
        0.5208333, 0.27083334,
        0.5625, 0.27083334,
        0.5625, 0.27083334,
        0.6041667, 0.27083334,
        0.6041667, 0.27083334,
        0.6458333, 0.27083334,
        0.6458333, 0.27083334,
        0.6875, 0.27083334,
        0.6875, 0.27083334,
        0.7291667, 0.27083334,
        0.7291667, 0.27083334,
        0.7708333, 0.27083334,
        0.7708333, 0.27083334,
        0.8125, 0.27083334,
        0.8125, 0.27083334,
        0.8541667, 0.27083334,
        0.8541667, 0.27083334,
        0.8958333, 0.27083334,
        0.8958333, 0.27083334,
        0.9375, 0.27083334,
        0.9375, 0.27083334,
        0.9791667, 0.27083334,
        0.9791667, 0.27083334,
        0.02083333, 0.3125,
        0.02083333, 0.3125,
        0.0625, 0.3125,
        0.0625, 0.3125,
        0.10416666, 0.3125,
        0.10416666, 0.3125,
        0.14583333, 0.3125,
        0.14583333, 0.3125,
        0.1875, 0.3125,
        0.1875, 0.3125,
        0.22916667, 0.3125,
        0.22916667, 0.3125,
        0.27083334, 0.3125,
        0.27083334, 0.3125,
        0.3125, 0.3125,
        0.3125, 0.3125,
        0.35416666, 0.3125,
        0.35416666, 0.3125,
        0.39583334, 0.3125,
        0.39583334, 0.3125,
        0.4375, 0.3125,
        0.4375, 0.3125,
        0.47916666, 0.3125,
        0.47916666, 0.3125,
        0.5208333, 0.3125,
        0.5208333, 0.3125,
        0.5625, 0.3125,
        0.5625, 0.3125,
        0.6041667, 0.3125,
        0.6041667, 0.3125,
        0.6458333, 0.3125,
        0.6458333, 0.3125,
        0.6875, 0.3125,
        0.6875, 0.3125,
        0.7291667, 0.3125,
        0.7291667, 0.3125,
        0.7708333, 0.3125,
        0.7708333, 0.3125,
        0.8125, 0.3125,
        0.8125, 0.3125,
        0.8541667, 0.3125,
        0.8541667, 0.3125,
        0.8958333, 0.3125,
        0.8958333, 0.3125,
        0.9375, 0.3125,
        0.9375, 0.3125,
        0.9791667, 0.3125,
        0.9791667, 0.3125,
        0.02083333, 0.35416666,
        0.02083333, 0.35416666,
        0.0625, 0.35416666,
        0.0625, 0.35416666,
        0.10416666, 0.35416666,
        0.10416666, 0.35416666,
        0.14583333, 0.35416666,
        0.14583333, 0.35416666,
        0.1875, 0.35416666,
        0.1875, 0.35416666,
        0.22916667, 0.35416666,
        0.22916667, 0.35416666,
        0.27083334, 0.35416666,
        0.27083334, 0.35416666,
        0.3125, 0.35416666,
        0.3125, 0.35416666,
        0.35416666, 0.35416666,
        0.35416666, 0.35416666,
        0.39583334, 0.35416666,
        0.39583334, 0.35416666,
        0.4375, 0.35416666,
        0.4375, 0.35416666,
        0.47916666, 0.35416666,
        0.47916666, 0.35416666,
        0.5208333, 0.35416666,
        0.5208333, 0.35416666,
        0.5625, 0.35416666,
        0.5625, 0.35416666,
        0.6041667, 0.35416666,
        0.6041667, 0.35416666,
        0.6458333, 0.35416666,
        0.6458333, 0.35416666,
        0.6875, 0.35416666,
        0.6875, 0.35416666,
        0.7291667, 0.35416666,
        0.7291667, 0.35416666,
        0.7708333, 0.35416666,
        0.7708333, 0.35416666,
        0.8125, 0.35416666,
        0.8125, 0.35416666,
        0.8541667, 0.35416666,
        0.8541667, 0.35416666,
        0.8958333, 0.35416666,
        0.8958333, 0.35416666,
        0.9375, 0.35416666,
        0.9375, 0.35416666,
        0.9791667, 0.35416666,
        0.9791667, 0.35416666,
        0.02083333, 0.39583334,
        0.02083333, 0.39583334,
        0.0625, 0.39583334,
        0.0625, 0.39583334,
        0.10416666, 0.39583334,
        0.10416666, 0.39583334,
        0.14583333, 0.39583334,
        0.14583333, 0.39583334,
        0.1875, 0.39583334,
        0.1875, 0.39583334,
        0.22916667, 0.39583334,
        0.22916667, 0.39583334,
        0.27083334, 0.39583334,
        0.27083334, 0.39583334,
        0.3125, 0.39583334,
        0.3125, 0.39583334,
        0.35416666, 0.39583334,
        0.35416666, 0.39583334,
        0.39583334, 0.39583334,
        0.39583334, 0.39583334,
        0.4375, 0.39583334,
        0.4375, 0.39583334,
        0.47916666, 0.39583334,
        0.47916666, 0.39583334,
        0.5208333, 0.39583334,
        0.5208333, 0.39583334,
        0.5625, 0.39583334,
        0.5625, 0.39583334,
        0.6041667, 0.39583334,
        0.6041667, 0.39583334,
        0.6458333, 0.39583334,
        0.6458333, 0.39583334,
        0.6875, 0.39583334,
        0.6875, 0.39583334,
        0.7291667, 0.39583334,
        0.7291667, 0.39583334,
        0.7708333, 0.39583334,
        0.7708333, 0.39583334,
        0.8125, 0.39583334,
        0.8125, 0.39583334,
        0.8541667, 0.39583334,
        0.8541667, 0.39583334,
        0.8958333, 0.39583334,
        0.8958333, 0.39583334,
        0.9375, 0.39583334,
        0.9375, 0.39583334,
        0.9791667, 0.39583334,
        0.9791667, 0.39583334,
        0.02083333, 0.4375,
        0.02083333, 0.4375,
        0.0625, 0.4375,
        0.0625, 0.4375,
        0.10416666, 0.4375,
        0.10416666, 0.4375,
        0.14583333, 0.4375,
        0.14583333, 0.4375,
        0.1875, 0.4375,
        0.1875, 0.4375,
        0.22916667, 0.4375,
        0.22916667, 0.4375,
        0.27083334, 0.4375,
        0.27083334, 0.4375,
        0.3125, 0.4375,
        0.3125, 0.4375,
        0.35416666, 0.4375,
        0.35416666, 0.4375,
        0.39583334, 0.4375,
        0.39583334, 0.4375,
        0.4375, 0.4375,
        0.4375, 0.4375,
        0.47916666, 0.4375,
        0.47916666, 0.4375,
        0.5208333, 0.4375,
        0.5208333, 0.4375,
        0.5625, 0.4375,
        0.5625, 0.4375,
        0.6041667, 0.4375,
        0.6041667, 0.4375,
        0.6458333, 0.4375,
        0.6458333, 0.4375,
        0.6875, 0.4375,
        0.6875, 0.4375,
        0.7291667, 0.4375,
        0.7291667, 0.4375,
        0.7708333, 0.4375,
        0.7708333, 0.4375,
        0.8125, 0.4375,
        0.8125, 0.4375,
        0.8541667, 0.4375,
        0.8541667, 0.4375,
        0.8958333, 0.4375,
        0.8958333, 0.4375,
        0.9375, 0.4375,
        0.9375, 0.4375,
        0.9791667, 0.4375,
        0.9791667, 0.4375,
        0.02083333, 0.47916666,
        0.02083333, 0.47916666,
        0.0625, 0.47916666,
        0.0625, 0.47916666,
        0.10416666, 0.47916666,
        0.10416666, 0.47916666,
        0.14583333, 0.47916666,
        0.14583333, 0.47916666,
        0.1875, 0.47916666,
        0.1875, 0.47916666,
        0.22916667, 0.47916666,
        0.22916667, 0.47916666,
        0.27083334, 0.47916666,
        0.27083334, 0.47916666,
        0.3125, 0.47916666,
        0.3125, 0.47916666,
        0.35416666, 0.47916666,
        0.35416666, 0.47916666,
        0.39583334, 0.47916666,
        0.39583334, 0.47916666,
        0.4375, 0.47916666,
        0.4375, 0.47916666,
        0.47916666, 0.47916666,
        0.47916666, 0.47916666,
        0.5208333, 0.47916666,
        0.5208333, 0.47916666,
        0.5625, 0.47916666,
        0.5625, 0.47916666,
        0.6041667, 0.47916666,
        0.6041667, 0.47916666,
        0.6458333, 0.47916666,
        0.6458333, 0.47916666,
        0.6875, 0.47916666,
        0.6875, 0.47916666,
        0.7291667, 0.47916666,
        0.7291667, 0.47916666,
        0.7708333, 0.47916666,
        0.7708333, 0.47916666,
        0.8125, 0.47916666,
        0.8125, 0.47916666,
        0.8541667, 0.47916666,
        0.8541667, 0.47916666,
        0.8958333, 0.47916666,
        0.8958333, 0.47916666,
        0.9375, 0.47916666,
        0.9375, 0.47916666,
        0.9791667, 0.47916666,
        0.9791667, 0.47916666,
        0.02083333, 0.5208333,
        0.02083333, 0.5208333,
        0.0625, 0.5208333,
        0.0625, 0.5208333,
        0.10416666, 0.5208333,
        0.10416666, 0.5208333,
        0.14583333, 0.5208333,
        0.14583333, 0.5208333,
        0.1875, 0.5208333,
        0.1875, 0.5208333,
        0.22916667, 0.5208333,
        0.22916667, 0.5208333,
        0.27083334, 0.5208333,
        0.27083334, 0.5208333,
        0.3125, 0.5208333,
        0.3125, 0.5208333,
        0.35416666, 0.5208333,
        0.35416666, 0.5208333,
        0.39583334, 0.5208333,
        0.39583334, 0.5208333,
        0.4375, 0.5208333,
        0.4375, 0.5208333,
        0.47916666, 0.5208333,
        0.47916666, 0.5208333,
        0.5208333, 0.5208333,
        0.5208333, 0.5208333,
        0.5625, 0.5208333,
        0.5625, 0.5208333,
        0.6041667, 0.5208333,
        0.6041667, 0.5208333,
        0.6458333, 0.5208333,
        0.6458333, 0.5208333,
        0.6875, 0.5208333,
        0.6875, 0.5208333,
        0.7291667, 0.5208333,
        0.7291667, 0.5208333,
        0.7708333, 0.5208333,
        0.7708333, 0.5208333,
        0.8125, 0.5208333,
        0.8125, 0.5208333,
        0.8541667, 0.5208333,
        0.8541667, 0.5208333,
        0.8958333, 0.5208333,
        0.8958333, 0.5208333,
        0.9375, 0.5208333,
        0.9375, 0.5208333,
        0.9791667, 0.5208333,
        0.9791667, 0.5208333,
        0.02083333, 0.5625,
        0.02083333, 0.5625,
        0.0625, 0.5625,
        0.0625, 0.5625,
        0.10416666, 0.5625,
        0.10416666, 0.5625,
        0.14583333, 0.5625,
        0.14583333, 0.5625,
        0.1875, 0.5625,
        0.1875, 0.5625,
        0.22916667, 0.5625,
        0.22916667, 0.5625,
        0.27083334, 0.5625,
        0.27083334, 0.5625,
        0.3125, 0.5625,
        0.3125, 0.5625,
        0.35416666, 0.5625,
        0.35416666, 0.5625,
        0.39583334, 0.5625,
        0.39583334, 0.5625,
        0.4375, 0.5625,
        0.4375, 0.5625,
        0.47916666, 0.5625,
        0.47916666, 0.5625,
        0.5208333, 0.5625,
        0.5208333, 0.5625,
        0.5625, 0.5625,
        0.5625, 0.5625,
        0.6041667, 0.5625,
        0.6041667, 0.5625,
        0.6458333, 0.5625,
        0.6458333, 0.5625,
        0.6875, 0.5625,
        0.6875, 0.5625,
        0.7291667, 0.5625,
        0.7291667, 0.5625,
        0.7708333, 0.5625,
        0.7708333, 0.5625,
        0.8125, 0.5625,
        0.8125, 0.5625,
        0.8541667, 0.5625,
        0.8541667, 0.5625,
        0.8958333, 0.5625,
        0.8958333, 0.5625,
        0.9375, 0.5625,
        0.9375, 0.5625,
        0.9791667, 0.5625,
        0.9791667, 0.5625,
        0.02083333, 0.6041667,
        0.02083333, 0.6041667,
        0.0625, 0.6041667,
        0.0625, 0.6041667,
        0.10416666, 0.6041667,
        0.10416666, 0.6041667,
        0.14583333, 0.6041667,
        0.14583333, 0.6041667,
        0.1875, 0.6041667,
        0.1875, 0.6041667,
        0.22916667, 0.6041667,
        0.22916667, 0.6041667,
        0.27083334, 0.6041667,
        0.27083334, 0.6041667,
        0.3125, 0.6041667,
        0.3125, 0.6041667,
        0.35416666, 0.6041667,
        0.35416666, 0.6041667,
        0.39583334, 0.6041667,
        0.39583334, 0.6041667,
        0.4375, 0.6041667,
        0.4375, 0.6041667,
        0.47916666, 0.6041667,
        0.47916666, 0.6041667,
        0.5208333, 0.6041667,
        0.5208333, 0.6041667,
        0.5625, 0.6041667,
        0.5625, 0.6041667,
        0.6041667, 0.6041667,
        0.6041667, 0.6041667,
        0.6458333, 0.6041667,
        0.6458333, 0.6041667,
        0.6875, 0.6041667,
        0.6875, 0.6041667,
        0.7291667, 0.6041667,
        0.7291667, 0.6041667,
        0.7708333, 0.6041667,
        0.7708333, 0.6041667,
        0.8125, 0.6041667,
        0.8125, 0.6041667,
        0.8541667, 0.6041667,
        0.8541667, 0.6041667,
        0.8958333, 0.6041667,
        0.8958333, 0.6041667,
        0.9375, 0.6041667,
        0.9375, 0.6041667,
        0.9791667, 0.6041667,
        0.9791667, 0.6041667,
        0.02083333, 0.6458333,
        0.02083333, 0.6458333,
        0.0625, 0.6458333,
        0.0625, 0.6458333,
        0.10416666, 0.6458333,
        0.10416666, 0.6458333,
        0.14583333, 0.6458333,
        0.14583333, 0.6458333,
        0.1875, 0.6458333,
        0.1875, 0.6458333,
        0.22916667, 0.6458333,
        0.22916667, 0.6458333,
        0.27083334, 0.6458333,
        0.27083334, 0.6458333,
        0.3125, 0.6458333,
        0.3125, 0.6458333,
        0.35416666, 0.6458333,
        0.35416666, 0.6458333,
        0.39583334, 0.6458333,
        0.39583334, 0.6458333,
        0.4375, 0.6458333,
        0.4375, 0.6458333,
        0.47916666, 0.6458333,
        0.47916666, 0.6458333,
        0.5208333, 0.6458333,
        0.5208333, 0.6458333,
        0.5625, 0.6458333,
        0.5625, 0.6458333,
        0.6041667, 0.6458333,
        0.6041667, 0.6458333,
        0.6458333, 0.6458333,
        0.6458333, 0.6458333,
        0.6875, 0.6458333,
        0.6875, 0.6458333,
        0.7291667, 0.6458333,
        0.7291667, 0.6458333,
        0.7708333, 0.6458333,
        0.7708333, 0.6458333,
        0.8125, 0.6458333,
        0.8125, 0.6458333,
        0.8541667, 0.6458333,
        0.8541667, 0.6458333,
        0.8958333, 0.6458333,
        0.8958333, 0.6458333,
        0.9375, 0.6458333,
        0.9375, 0.6458333,
        0.9791667, 0.6458333,
        0.9791667, 0.6458333,
        0.02083333, 0.6875,
        0.02083333, 0.6875,
        0.0625, 0.6875,
        0.0625, 0.6875,
        0.10416666, 0.6875,
        0.10416666, 0.6875,
        0.14583333, 0.6875,
        0.14583333, 0.6875,
        0.1875, 0.6875,
        0.1875, 0.6875,
        0.22916667, 0.6875,
        0.22916667, 0.6875,
        0.27083334, 0.6875,
        0.27083334, 0.6875,
        0.3125, 0.6875,
        0.3125, 0.6875,
        0.35416666, 0.6875,
        0.35416666, 0.6875,
        0.39583334, 0.6875,
        0.39583334, 0.6875,
        0.4375, 0.6875,
        0.4375, 0.6875,
        0.47916666, 0.6875,
        0.47916666, 0.6875,
        0.5208333, 0.6875,
        0.5208333, 0.6875,
        0.5625, 0.6875,
        0.5625, 0.6875,
        0.6041667, 0.6875,
        0.6041667, 0.6875,
        0.6458333, 0.6875,
        0.6458333, 0.6875,
        0.6875, 0.6875,
        0.6875, 0.6875,
        0.7291667, 0.6875,
        0.7291667, 0.6875,
        0.7708333, 0.6875,
        0.7708333, 0.6875,
        0.8125, 0.6875,
        0.8125, 0.6875,
        0.8541667, 0.6875,
        0.8541667, 0.6875,
        0.8958333, 0.6875,
        0.8958333, 0.6875,
        0.9375, 0.6875,
        0.9375, 0.6875,
        0.9791667, 0.6875,
        0.9791667, 0.6875,
        0.02083333, 0.7291667,
        0.02083333, 0.7291667,
        0.0625, 0.7291667,
        0.0625, 0.7291667,
        0.10416666, 0.7291667,
        0.10416666, 0.7291667,
        0.14583333, 0.7291667,
        0.14583333, 0.7291667,
        0.1875, 0.7291667,
        0.1875, 0.7291667,
        0.22916667, 0.7291667,
        0.22916667, 0.7291667,
        0.27083334, 0.7291667,
        0.27083334, 0.7291667,
        0.3125, 0.7291667,
        0.3125, 0.7291667,
        0.35416666, 0.7291667,
        0.35416666, 0.7291667,
        0.39583334, 0.7291667,
        0.39583334, 0.7291667,
        0.4375, 0.7291667,
        0.4375, 0.7291667,
        0.47916666, 0.7291667,
        0.47916666, 0.7291667,
        0.5208333, 0.7291667,
        0.5208333, 0.7291667,
        0.5625, 0.7291667,
        0.5625, 0.7291667,
        0.6041667, 0.7291667,
        0.6041667, 0.7291667,
        0.6458333, 0.7291667,
        0.6458333, 0.7291667,
        0.6875, 0.7291667,
        0.6875, 0.7291667,
        0.7291667, 0.7291667,
        0.7291667, 0.7291667,
        0.7708333, 0.7291667,
        0.7708333, 0.7291667,
        0.8125, 0.7291667,
        0.8125, 0.7291667,
        0.8541667, 0.7291667,
        0.8541667, 0.7291667,
        0.8958333, 0.7291667,
        0.8958333, 0.7291667,
        0.9375, 0.7291667,
        0.9375, 0.7291667,
        0.9791667, 0.7291667,
        0.9791667, 0.7291667,
        0.02083333, 0.7708333,
        0.02083333, 0.7708333,
        0.0625, 0.7708333,
        0.0625, 0.7708333,
        0.10416666, 0.7708333,
        0.10416666, 0.7708333,
        0.14583333, 0.7708333,
        0.14583333, 0.7708333,
        0.1875, 0.7708333,
        0.1875, 0.7708333,
        0.22916667, 0.7708333,
        0.22916667, 0.7708333,
        0.27083334, 0.7708333,
        0.27083334, 0.7708333,
        0.3125, 0.7708333,
        0.3125, 0.7708333,
        0.35416666, 0.7708333,
        0.35416666, 0.7708333,
        0.39583334, 0.7708333,
        0.39583334, 0.7708333,
        0.4375, 0.7708333,
        0.4375, 0.7708333,
        0.47916666, 0.7708333,
        0.47916666, 0.7708333,
        0.5208333, 0.7708333,
        0.5208333, 0.7708333,
        0.5625, 0.7708333,
        0.5625, 0.7708333,
        0.6041667, 0.7708333,
        0.6041667, 0.7708333,
        0.6458333, 0.7708333,
        0.6458333, 0.7708333,
        0.6875, 0.7708333,
        0.6875, 0.7708333,
        0.7291667, 0.7708333,
        0.7291667, 0.7708333,
        0.7708333, 0.7708333,
        0.7708333, 0.7708333,
        0.8125, 0.7708333,
        0.8125, 0.7708333,
        0.8541667, 0.7708333,
        0.8541667, 0.7708333,
        0.8958333, 0.7708333,
        0.8958333, 0.7708333,
        0.9375, 0.7708333,
        0.9375, 0.7708333,
        0.9791667, 0.7708333,
        0.9791667, 0.7708333,
        0.02083333, 0.8125,
        0.02083333, 0.8125,
        0.0625, 0.8125,
        0.0625, 0.8125,
        0.10416666, 0.8125,
        0.10416666, 0.8125,
        0.14583333, 0.8125,
        0.14583333, 0.8125,
        0.1875, 0.8125,
        0.1875, 0.8125,
        0.22916667, 0.8125,
        0.22916667, 0.8125,
        0.27083334, 0.8125,
        0.27083334, 0.8125,
        0.3125, 0.8125,
        0.3125, 0.8125,
        0.35416666, 0.8125,
        0.35416666, 0.8125,
        0.39583334, 0.8125,
        0.39583334, 0.8125,
        0.4375, 0.8125,
        0.4375, 0.8125,
        0.47916666, 0.8125,
        0.47916666, 0.8125,
        0.5208333, 0.8125,
        0.5208333, 0.8125,
        0.5625, 0.8125,
        0.5625, 0.8125,
        0.6041667, 0.8125,
        0.6041667, 0.8125,
        0.6458333, 0.8125,
        0.6458333, 0.8125,
        0.6875, 0.8125,
        0.6875, 0.8125,
        0.7291667, 0.8125,
        0.7291667, 0.8125,
        0.7708333, 0.8125,
        0.7708333, 0.8125,
        0.8125, 0.8125,
        0.8125, 0.8125,
        0.8541667, 0.8125,
        0.8541667, 0.8125,
        0.8958333, 0.8125,
        0.8958333, 0.8125,
        0.9375, 0.8125,
        0.9375, 0.8125,
        0.9791667, 0.8125,
        0.9791667, 0.8125,
        0.02083333, 0.8541667,
        0.02083333, 0.8541667,
        0.0625, 0.8541667,
        0.0625, 0.8541667,
        0.10416666, 0.8541667,
        0.10416666, 0.8541667,
        0.14583333, 0.8541667,
        0.14583333, 0.8541667,
        0.1875, 0.8541667,
        0.1875, 0.8541667,
        0.22916667, 0.8541667,
        0.22916667, 0.8541667,
        0.27083334, 0.8541667,
        0.27083334, 0.8541667,
        0.3125, 0.8541667,
        0.3125, 0.8541667,
        0.35416666, 0.8541667,
        0.35416666, 0.8541667,
        0.39583334, 0.8541667,
        0.39583334, 0.8541667,
        0.4375, 0.8541667,
        0.4375, 0.8541667,
        0.47916666, 0.8541667,
        0.47916666, 0.8541667,
        0.5208333, 0.8541667,
        0.5208333, 0.8541667,
        0.5625, 0.8541667,
        0.5625, 0.8541667,
        0.6041667, 0.8541667,
        0.6041667, 0.8541667,
        0.6458333, 0.8541667,
        0.6458333, 0.8541667,
        0.6875, 0.8541667,
        0.6875, 0.8541667,
        0.7291667, 0.8541667,
        0.7291667, 0.8541667,
        0.7708333, 0.8541667,
        0.7708333, 0.8541667,
        0.8125, 0.8541667,
        0.8125, 0.8541667,
        0.8541667, 0.8541667,
        0.8541667, 0.8541667,
        0.8958333, 0.8541667,
        0.8958333, 0.8541667,
        0.9375, 0.8541667,
        0.9375, 0.8541667,
        0.9791667, 0.8541667,
        0.9791667, 0.8541667,
        0.02083333, 0.8958333,
        0.02083333, 0.8958333,
        0.0625, 0.8958333,
        0.0625, 0.8958333,
        0.10416666, 0.8958333,
        0.10416666, 0.8958333,
        0.14583333, 0.8958333,
        0.14583333, 0.8958333,
        0.1875, 0.8958333,
        0.1875, 0.8958333,
        0.22916667, 0.8958333,
        0.22916667, 0.8958333,
        0.27083334, 0.8958333,
        0.27083334, 0.8958333,
        0.3125, 0.8958333,
        0.3125, 0.8958333,
        0.35416666, 0.8958333,
        0.35416666, 0.8958333,
        0.39583334, 0.8958333,
        0.39583334, 0.8958333,
        0.4375, 0.8958333,
        0.4375, 0.8958333,
        0.47916666, 0.8958333,
        0.47916666, 0.8958333,
        0.5208333, 0.8958333,
        0.5208333, 0.8958333,
        0.5625, 0.8958333,
        0.5625, 0.8958333,
        0.6041667, 0.8958333,
        0.6041667, 0.8958333,
        0.6458333, 0.8958333,
        0.6458333, 0.8958333,
        0.6875, 0.8958333,
        0.6875, 0.8958333,
        0.7291667, 0.8958333,
        0.7291667, 0.8958333,
        0.7708333, 0.8958333,
        0.7708333, 0.8958333,
        0.8125, 0.8958333,
        0.8125, 0.8958333,
        0.8541667, 0.8958333,
        0.8541667, 0.8958333,
        0.8958333, 0.8958333,
        0.8958333, 0.8958333,
        0.9375, 0.8958333,
        0.9375, 0.8958333,
        0.9791667, 0.8958333,
        0.9791667, 0.8958333,
        0.02083333, 0.9375,
        0.02083333, 0.9375,
        0.0625, 0.9375,
        0.0625, 0.9375,
        0.10416666, 0.9375,
        0.10416666, 0.9375,
        0.14583333, 0.9375,
        0.14583333, 0.9375,
        0.1875, 0.9375,
        0.1875, 0.9375,
        0.22916667, 0.9375,
        0.22916667, 0.9375,
        0.27083334, 0.9375,
        0.27083334, 0.9375,
        0.3125, 0.9375,
        0.3125, 0.9375,
        0.35416666, 0.9375,
        0.35416666, 0.9375,
        0.39583334, 0.9375,
        0.39583334, 0.9375,
        0.4375, 0.9375,
        0.4375, 0.9375,
        0.47916666, 0.9375,
        0.47916666, 0.9375,
        0.5208333, 0.9375,
        0.5208333, 0.9375,
        0.5625, 0.9375,
        0.5625, 0.9375,
        0.6041667, 0.9375,
        0.6041667, 0.9375,
        0.6458333, 0.9375,
        0.6458333, 0.9375,
        0.6875, 0.9375,
        0.6875, 0.9375,
        0.7291667, 0.9375,
        0.7291667, 0.9375,
        0.7708333, 0.9375,
        0.7708333, 0.9375,
        0.8125, 0.9375,
        0.8125, 0.9375,
        0.8541667, 0.9375,
        0.8541667, 0.9375,
        0.8958333, 0.9375,
        0.8958333, 0.9375,
        0.9375, 0.9375,
        0.9375, 0.9375,
        0.9791667, 0.9375,
        0.9791667, 0.9375,
        0.02083333, 0.9791667,
        0.02083333, 0.9791667,
        0.0625, 0.9791667,
        0.0625, 0.9791667,
        0.10416666, 0.9791667,
        0.10416666, 0.9791667,
        0.14583333, 0.9791667,
        0.14583333, 0.9791667,
        0.1875, 0.9791667,
        0.1875, 0.9791667,
        0.22916667, 0.9791667,
        0.22916667, 0.9791667,
        0.27083334, 0.9791667,
        0.27083334, 0.9791667,
        0.3125, 0.9791667,
        0.3125, 0.9791667,
        0.35416666, 0.9791667,
        0.35416666, 0.9791667,
        0.39583334, 0.9791667,
        0.39583334, 0.9791667,
        0.4375, 0.9791667,
        0.4375, 0.9791667,
        0.47916666, 0.9791667,
        0.47916666, 0.9791667,
        0.5208333, 0.9791667,
        0.5208333, 0.9791667,
        0.5625, 0.9791667,
        0.5625, 0.9791667,
        0.6041667, 0.9791667,
        0.6041667, 0.9791667,
        0.6458333, 0.9791667,
        0.6458333, 0.9791667,
        0.6875, 0.9791667,
        0.6875, 0.9791667,
        0.7291667, 0.9791667,
        0.7291667, 0.9791667,
        0.7708333, 0.9791667,
        0.7708333, 0.9791667,
        0.8125, 0.9791667,
        0.8125, 0.9791667,
        0.8541667, 0.9791667,
        0.8541667, 0.9791667,
        0.8958333, 0.9791667,
        0.8958333, 0.9791667,
        0.9375, 0.9791667,
        0.9375, 0.9791667,
        0.9791667, 0.9791667,
        0.9791667, 0.9791667,
        0.04166667, 0.04166667,
        0.04166667, 0.04166667,
        0.04166667, 0.04166667,
        0.04166667, 0.04166667,
        0.04166667, 0.04166667,
        0.04166667, 0.04166667,
        0.125, 0.04166667,
        0.125, 0.04166667,
        0.125, 0.04166667,
        0.125, 0.04166667,
        0.125, 0.04166667,
        0.125, 0.04166667,
        0.20833333, 0.04166667,
        0.20833333, 0.04166667,
        0.20833333, 0.04166667,
        0.20833333, 0.04166667,
        0.20833333, 0.04166667,
        0.20833333, 0.04166667,
        0.29166666, 0.04166667,
        0.29166666, 0.04166667,
        0.29166666, 0.04166667,
        0.29166666, 0.04166667,
        0.29166666, 0.04166667,
        0.29166666, 0.04166667,
        0.375, 0.04166667,
        0.375, 0.04166667,
        0.375, 0.04166667,
        0.375, 0.04166667,
        0.375, 0.04166667,
        0.375, 0.04166667,
        0.45833334, 0.04166667,
        0.45833334, 0.04166667,
        0.45833334, 0.04166667,
        0.45833334, 0.04166667,
        0.45833334, 0.04166667,
        0.45833334, 0.04166667,
        0.5416667, 0.04166667,
        0.5416667, 0.04166667,
        0.5416667, 0.04166667,
        0.5416667, 0.04166667,
        0.5416667, 0.04166667,
        0.5416667, 0.04166667,
        0.625, 0.04166667,
        0.625, 0.04166667,
        0.625, 0.04166667,
        0.625, 0.04166667,
        0.625, 0.04166667,
        0.625, 0.04166667,
        0.7083333, 0.04166667,
        0.7083333, 0.04166667,
        0.7083333, 0.04166667,
        0.7083333, 0.04166667,
        0.7083333, 0.04166667,
        0.7083333, 0.04166667,
        0.7916667, 0.04166667,
        0.7916667, 0.04166667,
        0.7916667, 0.04166667,
        0.7916667, 0.04166667,
        0.7916667, 0.04166667,
        0.7916667, 0.04166667,
        0.875, 0.04166667,
        0.875, 0.04166667,
        0.875, 0.04166667,
        0.875, 0.04166667,
        0.875, 0.04166667,
        0.875, 0.04166667,
        0.9583333, 0.04166667,
        0.9583333, 0.04166667,
        0.9583333, 0.04166667,
        0.9583333, 0.04166667,
        0.9583333, 0.04166667,
        0.9583333, 0.04166667,
        0.04166667, 0.125,
        0.04166667, 0.125,
        0.04166667, 0.125,
        0.04166667, 0.125,
        0.04166667, 0.125,
        0.04166667, 0.125,
        0.125, 0.125,
        0.125, 0.125,
        0.125, 0.125,
        0.125, 0.125,
        0.125, 0.125,
        0.125, 0.125,
        0.20833333, 0.125,
        0.20833333, 0.125,
        0.20833333, 0.125,
        0.20833333, 0.125,
        0.20833333, 0.125,
        0.20833333, 0.125,
        0.29166666, 0.125,
        0.29166666, 0.125,
        0.29166666, 0.125,
        0.29166666, 0.125,
        0.29166666, 0.125,
        0.29166666, 0.125,
        0.375, 0.125,
        0.375, 0.125,
        0.375, 0.125,
        0.375, 0.125,
        0.375, 0.125,
        0.375, 0.125,
        0.45833334, 0.125,
        0.45833334, 0.125,
        0.45833334, 0.125,
        0.45833334, 0.125,
        0.45833334, 0.125,
        0.45833334, 0.125,
        0.5416667, 0.125,
        0.5416667, 0.125,
        0.5416667, 0.125,
        0.5416667, 0.125,
        0.5416667, 0.125,
        0.5416667, 0.125,
        0.625, 0.125,
        0.625, 0.125,
        0.625, 0.125,
        0.625, 0.125,
        0.625, 0.125,
        0.625, 0.125,
        0.7083333, 0.125,
        0.7083333, 0.125,
        0.7083333, 0.125,
        0.7083333, 0.125,
        0.7083333, 0.125,
        0.7083333, 0.125,
        0.7916667, 0.125,
        0.7916667, 0.125,
        0.7916667, 0.125,
        0.7916667, 0.125,
        0.7916667, 0.125,
        0.7916667, 0.125,
        0.875, 0.125,
        0.875, 0.125,
        0.875, 0.125,
        0.875, 0.125,
        0.875, 0.125,
        0.875, 0.125,
        0.9583333, 0.125,
        0.9583333, 0.125,
        0.9583333, 0.125,
        0.9583333, 0.125,
        0.9583333, 0.125,
        0.9583333, 0.125,
        0.04166667, 0.20833333,
        0.04166667, 0.20833333,
        0.04166667, 0.20833333,
        0.04166667, 0.20833333,
        0.04166667, 0.20833333,
        0.04166667, 0.20833333,
        0.125, 0.20833333,
        0.125, 0.20833333,
        0.125, 0.20833333,
        0.125, 0.20833333,
        0.125, 0.20833333,
        0.125, 0.20833333,
        0.20833333, 0.20833333,
        0.20833333, 0.20833333,
        0.20833333, 0.20833333,
        0.20833333, 0.20833333,
        0.20833333, 0.20833333,
        0.20833333, 0.20833333,
        0.29166666, 0.20833333,
        0.29166666, 0.20833333,
        0.29166666, 0.20833333,
        0.29166666, 0.20833333,
        0.29166666, 0.20833333,
        0.29166666, 0.20833333,
        0.375, 0.20833333,
        0.375, 0.20833333,
        0.375, 0.20833333,
        0.375, 0.20833333,
        0.375, 0.20833333,
        0.375, 0.20833333,
        0.45833334, 0.20833333,
        0.45833334, 0.20833333,
        0.45833334, 0.20833333,
        0.45833334, 0.20833333,
        0.45833334, 0.20833333,
        0.45833334, 0.20833333,
        0.5416667, 0.20833333,
        0.5416667, 0.20833333,
        0.5416667, 0.20833333,
        0.5416667, 0.20833333,
        0.5416667, 0.20833333,
        0.5416667, 0.20833333,
        0.625, 0.20833333,
        0.625, 0.20833333,
        0.625, 0.20833333,
        0.625, 0.20833333,
        0.625, 0.20833333,
        0.625, 0.20833333,
        0.7083333, 0.20833333,
        0.7083333, 0.20833333,
        0.7083333, 0.20833333,
        0.7083333, 0.20833333,
        0.7083333, 0.20833333,
        0.7083333, 0.20833333,
        0.7916667, 0.20833333,
        0.7916667, 0.20833333,
        0.7916667, 0.20833333,
        0.7916667, 0.20833333,
        0.7916667, 0.20833333,
        0.7916667, 0.20833333,
        0.875, 0.20833333,
        0.875, 0.20833333,
        0.875, 0.20833333,
        0.875, 0.20833333,
        0.875, 0.20833333,
        0.875, 0.20833333,
        0.9583333, 0.20833333,
        0.9583333, 0.20833333,
        0.9583333, 0.20833333,
        0.9583333, 0.20833333,
        0.9583333, 0.20833333,
        0.9583333, 0.20833333,
        0.04166667, 0.29166666,
        0.04166667, 0.29166666,
        0.04166667, 0.29166666,
        0.04166667, 0.29166666,
        0.04166667, 0.29166666,
        0.04166667, 0.29166666,
        0.125, 0.29166666,
        0.125, 0.29166666,
        0.125, 0.29166666,
        0.125, 0.29166666,
        0.125, 0.29166666,
        0.125, 0.29166666,
        0.20833333, 0.29166666,
        0.20833333, 0.29166666,
        0.20833333, 0.29166666,
        0.20833333, 0.29166666,
        0.20833333, 0.29166666,
        0.20833333, 0.29166666,
        0.29166666, 0.29166666,
        0.29166666, 0.29166666,
        0.29166666, 0.29166666,
        0.29166666, 0.29166666,
        0.29166666, 0.29166666,
        0.29166666, 0.29166666,
        0.375, 0.29166666,
        0.375, 0.29166666,
        0.375, 0.29166666,
        0.375, 0.29166666,
        0.375, 0.29166666,
        0.375, 0.29166666,
        0.45833334, 0.29166666,
        0.45833334, 0.29166666,
        0.45833334, 0.29166666,
        0.45833334, 0.29166666,
        0.45833334, 0.29166666,
        0.45833334, 0.29166666,
        0.5416667, 0.29166666,
        0.5416667, 0.29166666,
        0.5416667, 0.29166666,
        0.5416667, 0.29166666,
        0.5416667, 0.29166666,
        0.5416667, 0.29166666,
        0.625, 0.29166666,
        0.625, 0.29166666,
        0.625, 0.29166666,
        0.625, 0.29166666,
        0.625, 0.29166666,
        0.625, 0.29166666,
        0.7083333, 0.29166666,
        0.7083333, 0.29166666,
        0.7083333, 0.29166666,
        0.7083333, 0.29166666,
        0.7083333, 0.29166666,
        0.7083333, 0.29166666,
        0.7916667, 0.29166666,
        0.7916667, 0.29166666,
        0.7916667, 0.29166666,
        0.7916667, 0.29166666,
        0.7916667, 0.29166666,
        0.7916667, 0.29166666,
        0.875, 0.29166666,
        0.875, 0.29166666,
        0.875, 0.29166666,
        0.875, 0.29166666,
        0.875, 0.29166666,
        0.875, 0.29166666,
        0.9583333, 0.29166666,
        0.9583333, 0.29166666,
        0.9583333, 0.29166666,
        0.9583333, 0.29166666,
        0.9583333, 0.29166666,
        0.9583333, 0.29166666,
        0.04166667, 0.375,
        0.04166667, 0.375,
        0.04166667, 0.375,
        0.04166667, 0.375,
        0.04166667, 0.375,
        0.04166667, 0.375,
        0.125, 0.375,
        0.125, 0.375,
        0.125, 0.375,
        0.125, 0.375,
        0.125, 0.375,
        0.125, 0.375,
        0.20833333, 0.375,
        0.20833333, 0.375,
        0.20833333, 0.375,
        0.20833333, 0.375,
        0.20833333, 0.375,
        0.20833333, 0.375,
        0.29166666, 0.375,
        0.29166666, 0.375,
        0.29166666, 0.375,
        0.29166666, 0.375,
        0.29166666, 0.375,
        0.29166666, 0.375,
        0.375, 0.375,
        0.375, 0.375,
        0.375, 0.375,
        0.375, 0.375,
        0.375, 0.375,
        0.375, 0.375,
        0.45833334, 0.375,
        0.45833334, 0.375,
        0.45833334, 0.375,
        0.45833334, 0.375,
        0.45833334, 0.375,
        0.45833334, 0.375,
        0.5416667, 0.375,
        0.5416667, 0.375,
        0.5416667, 0.375,
        0.5416667, 0.375,
        0.5416667, 0.375,
        0.5416667, 0.375,
        0.625, 0.375,
        0.625, 0.375,
        0.625, 0.375,
        0.625, 0.375,
        0.625, 0.375,
        0.625, 0.375,
        0.7083333, 0.375,
        0.7083333, 0.375,
        0.7083333, 0.375,
        0.7083333, 0.375,
        0.7083333, 0.375,
        0.7083333, 0.375,
        0.7916667, 0.375,
        0.7916667, 0.375,
        0.7916667, 0.375,
        0.7916667, 0.375,
        0.7916667, 0.375,
        0.7916667, 0.375,
        0.875, 0.375,
        0.875, 0.375,
        0.875, 0.375,
        0.875, 0.375,
        0.875, 0.375,
        0.875, 0.375,
        0.9583333, 0.375,
        0.9583333, 0.375,
        0.9583333, 0.375,
        0.9583333, 0.375,
        0.9583333, 0.375,
        0.9583333, 0.375,
        0.04166667, 0.45833334,
        0.04166667, 0.45833334,
        0.04166667, 0.45833334,
        0.04166667, 0.45833334,
        0.04166667, 0.45833334,
        0.04166667, 0.45833334,
        0.125, 0.45833334,
        0.125, 0.45833334,
        0.125, 0.45833334,
        0.125, 0.45833334,
        0.125, 0.45833334,
        0.125, 0.45833334,
        0.20833333, 0.45833334,
        0.20833333, 0.45833334,
        0.20833333, 0.45833334,
        0.20833333, 0.45833334,
        0.20833333, 0.45833334,
        0.20833333, 0.45833334,
        0.29166666, 0.45833334,
        0.29166666, 0.45833334,
        0.29166666, 0.45833334,
        0.29166666, 0.45833334,
        0.29166666, 0.45833334,
        0.29166666, 0.45833334,
        0.375, 0.45833334,
        0.375, 0.45833334,
        0.375, 0.45833334,
        0.375, 0.45833334,
        0.375, 0.45833334,
        0.375, 0.45833334,
        0.45833334, 0.45833334,
        0.45833334, 0.45833334,
        0.45833334, 0.45833334,
        0.45833334, 0.45833334,
        0.45833334, 0.45833334,
        0.45833334, 0.45833334,
        0.5416667, 0.45833334,
        0.5416667, 0.45833334,
        0.5416667, 0.45833334,
        0.5416667, 0.45833334,
        0.5416667, 0.45833334,
        0.5416667, 0.45833334,
        0.625, 0.45833334,
        0.625, 0.45833334,
        0.625, 0.45833334,
        0.625, 0.45833334,
        0.625, 0.45833334,
        0.625, 0.45833334,
        0.7083333, 0.45833334,
        0.7083333, 0.45833334,
        0.7083333, 0.45833334,
        0.7083333, 0.45833334,
        0.7083333, 0.45833334,
        0.7083333, 0.45833334,
        0.7916667, 0.45833334,
        0.7916667, 0.45833334,
        0.7916667, 0.45833334,
        0.7916667, 0.45833334,
        0.7916667, 0.45833334,
        0.7916667, 0.45833334,
        0.875, 0.45833334,
        0.875, 0.45833334,
        0.875, 0.45833334,
        0.875, 0.45833334,
        0.875, 0.45833334,
        0.875, 0.45833334,
        0.9583333, 0.45833334,
        0.9583333, 0.45833334,
        0.9583333, 0.45833334,
        0.9583333, 0.45833334,
        0.9583333, 0.45833334,
        0.9583333, 0.45833334,
        0.04166667, 0.5416667,
        0.04166667, 0.5416667,
        0.04166667, 0.5416667,
        0.04166667, 0.5416667,
        0.04166667, 0.5416667,
        0.04166667, 0.5416667,
        0.125, 0.5416667,
        0.125, 0.5416667,
        0.125, 0.5416667,
        0.125, 0.5416667,
        0.125, 0.5416667,
        0.125, 0.5416667,
        0.20833333, 0.5416667,
        0.20833333, 0.5416667,
        0.20833333, 0.5416667,
        0.20833333, 0.5416667,
        0.20833333, 0.5416667,
        0.20833333, 0.5416667,
        0.29166666, 0.5416667,
        0.29166666, 0.5416667,
        0.29166666, 0.5416667,
        0.29166666, 0.5416667,
        0.29166666, 0.5416667,
        0.29166666, 0.5416667,
        0.375, 0.5416667,
        0.375, 0.5416667,
        0.375, 0.5416667,
        0.375, 0.5416667,
        0.375, 0.5416667,
        0.375, 0.5416667,
        0.45833334, 0.5416667,
        0.45833334, 0.5416667,
        0.45833334, 0.5416667,
        0.45833334, 0.5416667,
        0.45833334, 0.5416667,
        0.45833334, 0.5416667,
        0.5416667, 0.5416667,
        0.5416667, 0.5416667,
        0.5416667, 0.5416667,
        0.5416667, 0.5416667,
        0.5416667, 0.5416667,
        0.5416667, 0.5416667,
        0.625, 0.5416667,
        0.625, 0.5416667,
        0.625, 0.5416667,
        0.625, 0.5416667,
        0.625, 0.5416667,
        0.625, 0.5416667,
        0.7083333, 0.5416667,
        0.7083333, 0.5416667,
        0.7083333, 0.5416667,
        0.7083333, 0.5416667,
        0.7083333, 0.5416667,
        0.7083333, 0.5416667,
        0.7916667, 0.5416667,
        0.7916667, 0.5416667,
        0.7916667, 0.5416667,
        0.7916667, 0.5416667,
        0.7916667, 0.5416667,
        0.7916667, 0.5416667,
        0.875, 0.5416667,
        0.875, 0.5416667,
        0.875, 0.5416667,
        0.875, 0.5416667,
        0.875, 0.5416667,
        0.875, 0.5416667,
        0.9583333, 0.5416667,
        0.9583333, 0.5416667,
        0.9583333, 0.5416667,
        0.9583333, 0.5416667,
        0.9583333, 0.5416667,
        0.9583333, 0.5416667,
        0.04166667, 0.625,
        0.04166667, 0.625,
        0.04166667, 0.625,
        0.04166667, 0.625,
        0.04166667, 0.625,
        0.04166667, 0.625,
        0.125, 0.625,
        0.125, 0.625,
        0.125, 0.625,
        0.125, 0.625,
        0.125, 0.625,
        0.125, 0.625,
        0.20833333, 0.625,
        0.20833333, 0.625,
        0.20833333, 0.625,
        0.20833333, 0.625,
        0.20833333, 0.625,
        0.20833333, 0.625,
        0.29166666, 0.625,
        0.29166666, 0.625,
        0.29166666, 0.625,
        0.29166666, 0.625,
        0.29166666, 0.625,
        0.29166666, 0.625,
        0.375, 0.625,
        0.375, 0.625,
        0.375, 0.625,
        0.375, 0.625,
        0.375, 0.625,
        0.375, 0.625,
        0.45833334, 0.625,
        0.45833334, 0.625,
        0.45833334, 0.625,
        0.45833334, 0.625,
        0.45833334, 0.625,
        0.45833334, 0.625,
        0.5416667, 0.625,
        0.5416667, 0.625,
        0.5416667, 0.625,
        0.5416667, 0.625,
        0.5416667, 0.625,
        0.5416667, 0.625,
        0.625, 0.625,
        0.625, 0.625,
        0.625, 0.625,
        0.625, 0.625,
        0.625, 0.625,
        0.625, 0.625,
        0.7083333, 0.625,
        0.7083333, 0.625,
        0.7083333, 0.625,
        0.7083333, 0.625,
        0.7083333, 0.625,
        0.7083333, 0.625,
        0.7916667, 0.625,
        0.7916667, 0.625,
        0.7916667, 0.625,
        0.7916667, 0.625,
        0.7916667, 0.625,
        0.7916667, 0.625,
        0.875, 0.625,
        0.875, 0.625,
        0.875, 0.625,
        0.875, 0.625,
        0.875, 0.625,
        0.875, 0.625,
        0.9583333, 0.625,
        0.9583333, 0.625,
        0.9583333, 0.625,
        0.9583333, 0.625,
        0.9583333, 0.625,
        0.9583333, 0.625,
        0.04166667, 0.7083333,
        0.04166667, 0.7083333,
        0.04166667, 0.7083333,
        0.04166667, 0.7083333,
        0.04166667, 0.7083333,
        0.04166667, 0.7083333,
        0.125, 0.7083333,
        0.125, 0.7083333,
        0.125, 0.7083333,
        0.125, 0.7083333,
        0.125, 0.7083333,
        0.125, 0.7083333,
        0.20833333, 0.7083333,
        0.20833333, 0.7083333,
        0.20833333, 0.7083333,
        0.20833333, 0.7083333,
        0.20833333, 0.7083333,
        0.20833333, 0.7083333,
        0.29166666, 0.7083333,
        0.29166666, 0.7083333,
        0.29166666, 0.7083333,
        0.29166666, 0.7083333,
        0.29166666, 0.7083333,
        0.29166666, 0.7083333,
        0.375, 0.7083333,
        0.375, 0.7083333,
        0.375, 0.7083333,
        0.375, 0.7083333,
        0.375, 0.7083333,
        0.375, 0.7083333,
        0.45833334, 0.7083333,
        0.45833334, 0.7083333,
        0.45833334, 0.7083333,
        0.45833334, 0.7083333,
        0.45833334, 0.7083333,
        0.45833334, 0.7083333,
        0.5416667, 0.7083333,
        0.5416667, 0.7083333,
        0.5416667, 0.7083333,
        0.5416667, 0.7083333,
        0.5416667, 0.7083333,
        0.5416667, 0.7083333,
        0.625, 0.7083333,
        0.625, 0.7083333,
        0.625, 0.7083333,
        0.625, 0.7083333,
        0.625, 0.7083333,
        0.625, 0.7083333,
        0.7083333, 0.7083333,
        0.7083333, 0.7083333,
        0.7083333, 0.7083333,
        0.7083333, 0.7083333,
        0.7083333, 0.7083333,
        0.7083333, 0.7083333,
        0.7916667, 0.7083333,
        0.7916667, 0.7083333,
        0.7916667, 0.7083333,
        0.7916667, 0.7083333,
        0.7916667, 0.7083333,
        0.7916667, 0.7083333,
        0.875, 0.7083333,
        0.875, 0.7083333,
        0.875, 0.7083333,
        0.875, 0.7083333,
        0.875, 0.7083333,
        0.875, 0.7083333,
        0.9583333, 0.7083333,
        0.9583333, 0.7083333,
        0.9583333, 0.7083333,
        0.9583333, 0.7083333,
        0.9583333, 0.7083333,
        0.9583333, 0.7083333,
        0.04166667, 0.7916667,
        0.04166667, 0.7916667,
        0.04166667, 0.7916667,
        0.04166667, 0.7916667,
        0.04166667, 0.7916667,
        0.04166667, 0.7916667,
        0.125, 0.7916667,
        0.125, 0.7916667,
        0.125, 0.7916667,
        0.125, 0.7916667,
        0.125, 0.7916667,
        0.125, 0.7916667,
        0.20833333, 0.7916667,
        0.20833333, 0.7916667,
        0.20833333, 0.7916667,
        0.20833333, 0.7916667,
        0.20833333, 0.7916667,
        0.20833333, 0.7916667,
        0.29166666, 0.7916667,
        0.29166666, 0.7916667,
        0.29166666, 0.7916667,
        0.29166666, 0.7916667,
        0.29166666, 0.7916667,
        0.29166666, 0.7916667,
        0.375, 0.7916667,
        0.375, 0.7916667,
        0.375, 0.7916667,
        0.375, 0.7916667,
        0.375, 0.7916667,
        0.375, 0.7916667,
        0.45833334, 0.7916667,
        0.45833334, 0.7916667,
        0.45833334, 0.7916667,
        0.45833334, 0.7916667,
        0.45833334, 0.7916667,
        0.45833334, 0.7916667,
        0.5416667, 0.7916667,
        0.5416667, 0.7916667,
        0.5416667, 0.7916667,
        0.5416667, 0.7916667,
        0.5416667, 0.7916667,
        0.5416667, 0.7916667,
        0.625, 0.7916667,
        0.625, 0.7916667,
        0.625, 0.7916667,
        0.625, 0.7916667,
        0.625, 0.7916667,
        0.625, 0.7916667,
        0.7083333, 0.7916667,
        0.7083333, 0.7916667,
        0.7083333, 0.7916667,
        0.7083333, 0.7916667,
        0.7083333, 0.7916667,
        0.7083333, 0.7916667,
        0.7916667, 0.7916667,
        0.7916667, 0.7916667,
        0.7916667, 0.7916667,
        0.7916667, 0.7916667,
        0.7916667, 0.7916667,
        0.7916667, 0.7916667,
        0.875, 0.7916667,
        0.875, 0.7916667,
        0.875, 0.7916667,
        0.875, 0.7916667,
        0.875, 0.7916667,
        0.875, 0.7916667,
        0.9583333, 0.7916667,
        0.9583333, 0.7916667,
        0.9583333, 0.7916667,
        0.9583333, 0.7916667,
        0.9583333, 0.7916667,
        0.9583333, 0.7916667,
        0.04166667, 0.875,
        0.04166667, 0.875,
        0.04166667, 0.875,
        0.04166667, 0.875,
        0.04166667, 0.875,
        0.04166667, 0.875,
        0.125, 0.875,
        0.125, 0.875,
        0.125, 0.875,
        0.125, 0.875,
        0.125, 0.875,
        0.125, 0.875,
        0.20833333, 0.875,
        0.20833333, 0.875,
        0.20833333, 0.875,
        0.20833333, 0.875,
        0.20833333, 0.875,
        0.20833333, 0.875,
        0.29166666, 0.875,
        0.29166666, 0.875,
        0.29166666, 0.875,
        0.29166666, 0.875,
        0.29166666, 0.875,
        0.29166666, 0.875,
        0.375, 0.875,
        0.375, 0.875,
        0.375, 0.875,
        0.375, 0.875,
        0.375, 0.875,
        0.375, 0.875,
        0.45833334, 0.875,
        0.45833334, 0.875,
        0.45833334, 0.875,
        0.45833334, 0.875,
        0.45833334, 0.875,
        0.45833334, 0.875,
        0.5416667, 0.875,
        0.5416667, 0.875,
        0.5416667, 0.875,
        0.5416667, 0.875,
        0.5416667, 0.875,
        0.5416667, 0.875,
        0.625, 0.875,
        0.625, 0.875,
        0.625, 0.875,
        0.625, 0.875,
        0.625, 0.875,
        0.625, 0.875,
        0.7083333, 0.875,
        0.7083333, 0.875,
        0.7083333, 0.875,
        0.7083333, 0.875,
        0.7083333, 0.875,
        0.7083333, 0.875,
        0.7916667, 0.875,
        0.7916667, 0.875,
        0.7916667, 0.875,
        0.7916667, 0.875,
        0.7916667, 0.875,
        0.7916667, 0.875,
        0.875, 0.875,
        0.875, 0.875,
        0.875, 0.875,
        0.875, 0.875,
        0.875, 0.875,
        0.875, 0.875,
        0.9583333, 0.875,
        0.9583333, 0.875,
        0.9583333, 0.875,
        0.9583333, 0.875,
        0.9583333, 0.875,
        0.9583333, 0.875,
        0.04166667, 0.9583333,
        0.04166667, 0.9583333,
        0.04166667, 0.9583333,
        0.04166667, 0.9583333,
        0.04166667, 0.9583333,
        0.04166667, 0.9583333,
        0.125, 0.9583333,
        0.125, 0.9583333,
        0.125, 0.9583333,
        0.125, 0.9583333,
        0.125, 0.9583333,
        0.125, 0.9583333,
        0.20833333, 0.9583333,
        0.20833333, 0.9583333,
        0.20833333, 0.9583333,
        0.20833333, 0.9583333,
        0.20833333, 0.9583333,
        0.20833333, 0.9583333,
        0.29166666, 0.9583333,
        0.29166666, 0.9583333,
        0.29166666, 0.9583333,
        0.29166666, 0.9583333,
        0.29166666, 0.9583333,
        0.29166666, 0.9583333,
        0.375, 0.9583333,
        0.375, 0.9583333,
        0.375, 0.9583333,
        0.375, 0.9583333,
        0.375, 0.9583333,
        0.375, 0.9583333,
        0.45833334, 0.9583333,
        0.45833334, 0.9583333,
        0.45833334, 0.9583333,
        0.45833334, 0.9583333,
        0.45833334, 0.9583333,
        0.45833334, 0.9583333,
        0.5416667, 0.9583333,
        0.5416667, 0.9583333,
        0.5416667, 0.9583333,
        0.5416667, 0.9583333,
        0.5416667, 0.9583333,
        0.5416667, 0.9583333,
        0.625, 0.9583333,
        0.625, 0.9583333,
        0.625, 0.9583333,
        0.625, 0.9583333,
        0.625, 0.9583333,
        0.625, 0.9583333,
        0.7083333, 0.9583333,
        0.7083333, 0.9583333,
        0.7083333, 0.9583333,
        0.7083333, 0.9583333,
        0.7083333, 0.9583333,
        0.7083333, 0.9583333,
        0.7916667, 0.9583333,
        0.7916667, 0.9583333,
        0.7916667, 0.9583333,
        0.7916667, 0.9583333,
        0.7916667, 0.9583333,
        0.7916667, 0.9583333,
        0.875, 0.9583333,
        0.875, 0.9583333,
        0.875, 0.9583333,
        0.875, 0.9583333,
        0.875, 0.9583333,
        0.875, 0.9583333,
        0.9583333, 0.9583333,
        0.9583333, 0.9583333,
        0.9583333, 0.9583333,
        0.9583333, 0.9583333,
        0.9583333, 0.9583333,
        0.9583333, 0.9583333);
}
