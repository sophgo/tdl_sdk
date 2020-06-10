#pragma once
#include "linear.h"
#include <opencv2/opencv.hpp>
using namespace std;

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

class LinearParam{

public:
    /**
     * Default solver 1.
     */
	static parameter construct_param(){
		return parameter{ L2R_L2LOSS_SVC_DUAL, 0.1, 1, 0, NULL, NULL, 0.1 };
	}
    /*
     * User-defined parameters. Use this only when you know
     * what you are doing.
     */
	static parameter construct_param(int solver_type,
		double eps,
		double C,
		int nr_weight,
		int *weight_label,
		double* weight,
		double p){
		return parameter{ solver_type, eps, C, nr_weight, weight_label, weight, p };
	}
    
    /*
     * Only choose solver type and the rest is set for you.
     * Recommand: choose solver 1 first. If warning appears,
     * choose solver 2. If you want to have probability output, 
     * choose solver 0 or 7.
     */
    static parameter construct_param(int solver_type){
        double C = 1;
        double p = 0.1;
        double eps = INF;
        int nr_weight = 0;
        int *weight_label = nullptr;
        double* weight = nullptr;
        
        switch (solver_type){
            case L2R_LR:
            case L2R_L2LOSS_SVC:
                eps = 0.01;
                break;
            case L2R_L2LOSS_SVR:
                eps = 0.001;
                break;
            case L2R_L2LOSS_SVC_DUAL:
            case L2R_L1LOSS_SVC_DUAL:
            case MCSVM_CS:
            case L2R_LR_DUAL:
                eps = 0.1;
                break;
            case L1R_L2LOSS_SVC:
            case L1R_LR:
                eps = 0.01;
                break;
            case L2R_L1LOSS_SVR_DUAL:
            case L2R_L2LOSS_SVR_DUAL:
                eps = 0.1;
                break;
        }
        
        return parameter{ solver_type, eps, C, nr_weight, weight_label, weight, p };
    }
    
    static void destroy_param(parameter *param){
        if(param){
            ::destroy_param(param);
            param = nullptr;
        }
    }
    
    static void exit_with_help()
    {
        printf(
               "Usage: train [options] training_set_file [model_file]\n"
               "options:\n"
               "-s type : set type of solver (default 1)\n"
               "  for multi-class classification\n"
               "	 0 -- L2-regularized logistic regression (primal)\n"
               "	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
               "	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
               "	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
               "	 4 -- support vector classification by Crammer and Singer\n"
               "	 5 -- L1-regularized L2-loss support vector classification\n"
               "	 6 -- L1-regularized logistic regression\n"
               "	 7 -- L2-regularized logistic regression (dual)\n"
               "  for regression\n"
               "	11 -- L2-regularized L2-loss support vector regression (primal)\n"
               "	12 -- L2-regularized L2-loss support vector regression (dual)\n"
               "	13 -- L2-regularized L1-loss support vector regression (dual)\n"
               "-c cost : set the parameter C (default 1)\n"
               "-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
               "-e epsilon : set tolerance of termination criterion\n"
               "	-s 0 and 2\n"
               "		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
               "		where f is the primal function and pos/neg are # of\n"
               "		positive/negative data (default 0.01)\n"
               "	-s 11\n"
               "		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.001)\n"
               "	-s 1, 3, 4, and 7\n"
               "		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
               "	-s 5 and 6\n"
               "		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
               "		where f is the primal function (default 0.01)\n"
               "	-s 12 and 13\n"
               "		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
               "		where f is the dual function (default 0.1)\n"
               "-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
               "-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
               "-v n: n-fold cross validation mode\n"
               "-q : quiet mode (no outputs)\n"
               );
        exit(1);
    }

};


class LibLinear{
    
private:
	struct parameter _param;
	struct problem _prob;
	struct model *_model;
	feature_node *_sample;

	/**
	 * prob: l->number of samples(labels), n->number of features, bias->-1 by default,
	 * x->pointer to feature data, y->pointer to label data.
	 */
	void convert_train_data(const cv::Mat &FeatureMat, const cv::Mat &LabelMat, problem &prob){
		if (FeatureMat.type() != CV_32FC1 || LabelMat.type() != CV_32FC1)
			cerr << "sorry, feature mat and label mat should be float\n";
		prob.bias = -1; // in this version, no bias term is considered.
		prob.l = FeatureMat.rows;
		prob.n = FeatureMat.cols;
		prob.x = Malloc(feature_node *, prob.l);
		prob.y = Malloc(double, prob.l);
		int feature_num = prob.n + 1;
		_sample = Malloc(feature_node, prob.l*feature_num);
		for (int i = 0; i < prob.l; i++)
		{
			for (int j = 0; j < prob.n; j++)
			{
				_sample[i*feature_num+j].index = j+1;
				_sample[i*feature_num+j].value = FeatureMat.at<float>(i, j);
			}
			_sample[(i+1)*feature_num-1].index = -1;
			prob.x[i] = &_sample[i*feature_num];
		}
		for (int i = 0; i < LabelMat.rows; i++)
			prob.y[i] = LabelMat.at<float>(i);
	}
    
    void convert_train_data(const float *data, const float *label,
                            int nSamples, int nFeatures,
                            problem &prob){
        prob.bias = -1;
        prob.l = nSamples;
        prob.n = nFeatures;
        prob.x = Malloc(feature_node *, prob.l);
        prob.y = Malloc(double, prob.l);
        int feature_num = prob.n + 1;
        _sample = Malloc(feature_node, prob.l*feature_num);
        for (int i = 0; i < prob.l; i++)
        {
            for (int j = 0; j < prob.n; j++)
            {
                _sample[i*feature_num+j].index = j+1;
                _sample[i*feature_num+j].value = data[i*prob.n+j];
            }
            _sample[(i+1)*feature_num-1].index = -1;
            prob.x[i] = &_sample[i*feature_num];
        }
        for (int i = 0; i < prob.l; i++)
            prob.y[i] = label[i];
    }
    
	/**
	 * one sample each call.
	 */
	void convert_test_data(const cv::Mat &SampleMat, feature_node **x){
		int feature_num = SampleMat.cols + 1;
		*x = Malloc(feature_node, feature_num);
		for (int i = 0; i < SampleMat.cols; i++){
			(*x)[i].index = i+1;
			(*x)[i].value = SampleMat.at<float>(i);
		}
		(*x)[feature_num - 1].index = -1;
	}
    
    void convert_test_data(const float *sample, int nFeatures,
                           feature_node **x){
        int feature_num = nFeatures + 1;
        *x = Malloc(feature_node, feature_num);
        for (int i = 0; i < nFeatures; i++){
            (*x)[i].index = i+1;
            (*x)[i].value = sample[i];
        }
        (*x)[feature_num - 1].index = -1;
    }


public:
    
	~LibLinear(){
            //destroy_param(&_param);
        if(_model)
            free_and_destroy_model(&_model);
		if(_prob.y)
            free(_prob.y);
		if(_prob.x)
            free(_prob.x);
		if(_sample)
            free(_sample);
        //cout<<"destructor is called\n";
	}
    
	LibLinear(){
		_param = parameter();
		_prob = problem();
		_model = nullptr;
		_sample = nullptr;
	}
    
    void release(){
        if(_model){
            free_and_destroy_model(&_model);
            _model = nullptr;
        }
        if(_prob.x){
            free(_prob.x);
            _prob.x = nullptr;
        }
        if(_prob.y){
            free(_prob.y);
            _prob.y = nullptr;
        }
        if(_sample){
            free(_sample);
            _sample = nullptr;
        }
    }
    
	void save_model(string model_file_name){
        ::save_model(model_file_name.c_str(), _model);
	}
    
	/**
	 * FeatureMat: M samples * N features.
	 * LabelMat: N lables * 1.
	 * param: liblinear parameters.
	 */
	void train(cv::Mat &FeatureMat,
		cv::Mat &LabelMat,
		parameter &param){
		_param = param;
		convert_train_data(FeatureMat, LabelMat, _prob);
        
        // check feasiblity of _model, _prob and _param
        if(_model){
            free_and_destroy_model(&_model);
        }
        if(NULL != ::check_parameter(&_prob, &_param))
            LinearParam::exit_with_help();
        
        // time and train
        //const int64 s1 = cv::getTickCount();
        _model = ::train(&_prob, &_param);
        //const int64 s2 = cv::getTickCount();
        //fprintf(stdout, "train finished! Use %8d s\n",
                //static_cast<int>((s2 - s1) / cv::getTickFrequency()));
        
        // clear internal data right after model is trained.
        // as they are useless now.
        if(_sample){
            free(_prob.x);
            free(_prob.y);
            free(_sample);
            _prob.x = nullptr;
            _prob.y = nullptr;
            _sample = nullptr;
        }
	}
    
    void train(const float *data, const float *label,
               int nSamples, int nFeatures, parameter &param){
        _param = param;
        convert_train_data(data, label, nSamples, nFeatures, _prob);
        // check feasiblity of _model, _prob and _param
        if(_model){
            free_and_destroy_model(&_model);
        }
        if(NULL != ::check_parameter(&_prob, &_param))
            LinearParam::exit_with_help();
        
        // time and train
        //const int64 s1 = cv::getTickCount();
        _model = ::train(&_prob, &_param);
        //const int64 s2 = cv::getTickCount();
        //fprintf(stdout, "train finished! Use %8d s\n",
                //static_cast<int>((s2 - s1) / cv::getTickFrequency()));
        
        // clear internal data right after model is trained.
        // as they are useless now.
        if(_sample){
            free(_prob.x);
            free(_prob.y);
            free(_sample);
            _prob.x = nullptr;
            _prob.y = nullptr;
            _sample = nullptr;
        }
    }
    
    /**
     * SampleMat contains only one sample.
     */
    double predict(cv::Mat &SampleMat){
        assert(SampleMat.rows == 1);
        feature_node *x = nullptr;
        convert_test_data(SampleMat, &x);
        double out = ::predict(_model, x);
        free(x);
        return out;
    }
    
    /**
     * SamplesMat contains multiple samples in multiple rows.
     * OutputMat is the predicted labels for each sample.
     */
	void predict(cv::Mat &SamplesMat, cv::Mat &OutputMat){
		feature_node *x = nullptr;
		OutputMat.create(SamplesMat.rows, 1, CV_32FC1);
		for (int i = 0; i < SamplesMat.rows; i++){
			convert_test_data(SamplesMat.row(i), &x);
            double out = ::predict(_model, x);
			OutputMat.at<float>(i) = out;
			free(x);
		}
	}
    
    /*
     * ValueMat contains Sigma(w*x)
     */
    double predict_values(cv::Mat &SampleMat, cv::Mat &ValueMat){
        assert(SampleMat.rows == 1);
        feature_node *x = nullptr;
        convert_test_data(SampleMat, &x);
        
        int nr_class=_model->nr_class;
        int nr_w;
        if(nr_class==2 && _model->param.solver_type != MCSVM_CS)
            nr_w = 1;
        else
            nr_w = nr_class;
        ValueMat.create(nr_w, 1, CV_32FC1);
        double *v = Malloc(double, nr_w);
        double out = ::predict_values(_model, x, v);
        for (int i = 0; i < nr_w; i++) {
            ValueMat.at<float>(i) = v[i];
        }
        free(x);
        free(v);
        return out;
    }
    
    double predict_values(float *sample, int nFeatures, float *value){
        
        feature_node *x = nullptr;
        convert_test_data(sample, nFeatures, &x);
        assert(_model->nr_class == 2);
        
        double pval = 0;
        double out = ::predict_values(_model, x, &pval);
        *value = pval;
        free(x);
        return out;
    }
    
    double predict_probabilities(cv::Mat &SampleMat, cv::Mat &ProbMat){
        assert(SampleMat.rows == 1);
        feature_node *x = nullptr;
        convert_test_data(SampleMat, &x);
        
        int nr_class=_model->nr_class;
        int nr_w;
        if(nr_class==2 && _model->param.solver_type != MCSVM_CS)
            nr_w = 1;
        else
            nr_w = nr_class;
        ProbMat.create(nr_w, 1, CV_32FC1);
        double *p = Malloc(double, nr_w);
        double out = ::predict_probability(_model, x, p);
        for (int i = 0; i < nr_w; i++) {
            ProbMat.at<float>(i) = p[i];
        }
        
        free(x);
        free(p);
        if(!check_probability_model(_model)){
            cout<<"This solver does not support probability output, please use Solver 0 or 7 instead\n";
        }
        return out;
    }
    
    void load_model(string model_file_name){
        if(_model){
            free_and_destroy_model(&_model);
        }
        _model = ::load_model(model_file_name.c_str());
    }
    
    void get_w(double *val){
        memcpy(val, _model->w, sizeof(double)*_model->nr_feature);
    }
};