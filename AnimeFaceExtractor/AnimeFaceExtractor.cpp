#include <iostream>
#include <sstream>
#include <iomanip>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>

using namespace std;
using namespace cv;

Ptr<CascadeClassifier> face_detector;

void extractFacesFromMovie(const string &inputMovieName,
	const string &outputFolder, const int beginId, int *buf_endId, const int frame_interval = 10);

int main() {
	const string inputFileName = "./input.mp4";
	const string outputFolderName = "./face";
	face_detector = new CascadeClassifier("./lbpcascade_animeface.xml");

	int face_id = 1;
	extractFacesFromMovie(inputFileName, outputFolderName, face_id, &face_id, 10);

	return 0;
}

void extractFacesFromMovie(const string &inputMovieName, const string &outputFolder,
	const int beginId, int *buf_endId, const int frame_interval) {
	
	Ptr<VideoCapture> movie(new VideoCapture(inputMovieName));
	assert(movie->isOpened());
	(*buf_endId) = beginId;
	for (int frame_id = 1; ; frame_id++) {
		// フレームを読み込む
		Mat frame;
		(*movie) >> frame;
		if (frame.empty()) { break; }
		if (frame_id % frame_interval != 0) { continue; }
		cout << to_string(frame_id);
		// グレースケール化 & ヒストグラム平坦化
		Mat_<uchar> frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		// 顔を検出する
		vector<Rect> faces;
		face_detector->detectMultiScale(
			frame_gray, faces, 1.1, 5, 0, Size(24, 24));
		if (faces.size() == 0) {
			cout << " ";
			continue;
		}
		cout << "(" << to_string(faces.size()) << ") ";

		// 64x64にリサイズ、色味を正規化、保存
		for (auto itr = faces.begin(); itr != faces.end(); ++itr) {
			Mat face = frame(*itr);
			Mat face_resized;
			resize(face, face_resized, Size(64, 64));
			Mat face_norm;
			normalize(face_resized, face_norm, 0, 255, NORM_MINMAX, CV_8UC3);

			ostringstream filename;
			filename << outputFolder << "/" <<
				setw(8) << setfill('0') << (*buf_endId)++ << ".png";
			assert(imwrite(filename.str(), face_norm));
		}
	}
	cout << "\nMovie Finished. : " << inputMovieName << endl << endl;
	return;
}