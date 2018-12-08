#include <iostream>
#include <sstream>
#include <memory>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

using namespace std;
using namespace cv;

Ptr<Mat_<float>> getFeature(const Mat_<Vec3b> &src);

int main() {
	const string testFolderName = "./face";
	const string resultFolderName = "./result";

	// �e���v���[�g�̓����ʂ𓾂�
	Ptr<Mat_<float>> feature_templates;
	for (int cate_id = 1;; cate_id++) {
		ostringstream fileName;
		fileName << resultFolderName << "/" << cate_id << "/" <<
			"template.png";
		Mat_<Vec3b> src = imread(fileName.str());
		if (src.empty()) { break; }

		if (cate_id == 1) { feature_templates = getFeature(src); }
		else { vconcat(*feature_templates, *getFeature(src), *feature_templates); }
	}

	// �e�X�g�f�[�^�̕��ނ��s��
	for (int test_id = 1;; test_id++) {

		// �e�X�g�f�[�^��ǂݍ��݁A�����ʂ𓾂�
		ostringstream fileName;
		fileName << testFolderName << "/" <<
			setw(8) << setfill('0') << test_id << ".png";
		Mat_<Vec3b> src = imread(fileName.str());
		if (src.empty()) { break; }

		Ptr<Mat_<float>> feature_test = getFeature(src);

		// �����ʊԂ̋������v�Z���A�}�b�`���O���s��
		BFMatcher matcher(NORM_L1);
		vector<DMatch> matches;
		matcher.match(*feature_test, *feature_templates, matches);

		// �}�b�`���O���ʂɑΉ�����t�H���_�Ƀe�X�g�摜��ۑ�����
		const int folder_number = (matches[0].trainIdx + 1);
		ostringstream newFileName;
		newFileName << resultFolderName << "/" << folder_number << "/"
			<< setw(8) << setfill('0') << test_id << ".png";
		imwrite(newFileName.str(), src);
		cout << fileName.str() << " -> " << to_string(folder_number) << endl;
	}

	return 0;
}

// �摜����AHue�̊�����p����256�����̓����ʂ𓾂�B
Ptr<Mat_<float>> getFeature(const Mat_<Vec3b> &src) {

	Ptr<Mat_<float>> feature(new Mat_<float>(1, 256, 0.0));

	// �摜��HSV�ɕϊ�����
	Mat_<Vec3b> src_hsv;
	cvtColor(src, src_hsv, CV_BGR2HSV);

	// Hue�̊���������ʂƂ���
	for (int y = 0; y < src_hsv.rows; y++) {
		for (int x = 0; x < src_hsv.cols; x++) {
			const uchar hue = src_hsv.at<Vec3b>(y, x)[0];
			feature->at<float>(0, hue)++;
		}
	}
	(*feature) /= (src_hsv.rows * src_hsv.cols);

	return feature;
}
