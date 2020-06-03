
#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <random>

#include <ros/ros.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TwistStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Float64MultiArray.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

//q as matrix
Eigen::Matrix4f getqMat(Eigen::Vector4f q)
{
	Eigen::Matrix4f qMat;
	qMat << q(0), -q(1), -q(2), -q(3),
			q(1),  q(0), -q(3),  q(2),
			q(2),  q(3),  q(0), -q(1),
			q(3), -q(2),  q(1),  q(0);
	return qMat;
}

//q inverse
Eigen::Vector4f getqInv(Eigen::Vector4f q)
{
	Eigen::Vector4f qInv;
	qInv << q(0), -q(1), -q(2), -q(3);
	return qInv;
}

void wallCheck(Eigen::Vector2f& position, Eigen::Vector2f& velocity, Eigen::Vector4f wall, float dt)
{
	Eigen::Vector2f positionNew = position + velocity*dt;

	// check first position lower bound
	if ((positionNew(0) < wall(0)) && (velocity(0) < 0))
	{
		position(0) = wall(0);
		velocity(0) = 0;
	}
	// check first position upper bound
	if ((positionNew(0) > wall(1)) && (velocity(0) > 0))
	{
		position(0) = wall(1);
		velocity(0) = 0;
	}
	// check second position lower bound
	if ((positionNew(1) < wall(2)) && (velocity(1) < 0))
	{
		position(1) = wall(2);
		velocity(1) = 0;
	}
	// check second position upper bound
	if ((positionNew(1) > wall(3)) && (velocity(1) > 0))
	{
		position(1) = wall(3);
		velocity(1) = 0;
	}
}

// sheep object, each handles its own data storage and publishing and
// shares it with the bear when requested through get and set functions
class Sheep
{
	const std::string sheepName;         // sheep bebop name
	int sheepNumber;                     // indicates which sheep this is
	ros::NodeHandle nh;                  // nodehandle
	ros::Subscriber xSub;                // sheep position subscriber, subscribe to pose message
	ros::Subscriber xDotSub;             // sheep linear velocity subscriber, subscribe to twist message
	ros::Timer minEigTimer;              // timer to calculate minimum eigenvalues
	Eigen::Vector2f y;                   // bear position
	Eigen::Vector2f x;                   // sheep position
	Eigen::Vector2f xDotMeas;            // sheep measured velocity
	Eigen::Vector2f xg;                  // goal location for the sheep
	Eigen::Vector2f xF;                  // final sheep position
	bool pushToGoal;					 // boolean to indicate the sheep should be pushed to goal
	bool firstMocap;                     // indicates first mocap has been received by the sheep
	bool chased;                         // indicates if this sheep is being chased
	float K1;                           // constant gain
	float Gamma;
	float kcl;                          // CL gain
	float Deltat;                       // buffer window
	Eigen::VectorXf Y;                   // Y value
	std::deque<Eigen::Vector2f> xBuffer; // buffer for the sheep position
	std::deque<Eigen::VectorXf> YBuffer; // buffer for the sheep Y
	std::deque<ros::Time> timeBuffer;    // buffer for the time
	float unchasedTime;                 // total chased time
	float chasedTime;                   // total unchased time
	ros::Time timeLast;                  // last mocap pose time
	float height;                       // height for the sheep to hover at
	bool killSheep;						 // indicates to kill the sheep
	bool dataSaved;
	bool saveData;
	std::string runNumber;
	Eigen::MatrixXf M;                   // means for the weights of the basis functions
	float sm2;                          // variance to choose the means
	float s2;							 // variance for the weights
	int L1;                              // number of neurons
	Eigen::Vector3f yP;
	Eigen::Vector4f yQ;
	int polyOrder;
	ros::Time firstTime;
	Eigen::MatrixXf YTYSumj;
	Eigen::MatrixXf YTDxSumj;
	Eigen::MatrixXf YTYSumj1;
	Eigen::MatrixXf YTDxSumj1;
	float minEigj;
	float minEigj1;
	int N;
	float lambda;
	std::vector<Eigen::Vector2f> xData;  // x data for the save
	std::vector<float> timeData;     // time data for the save
	std::vector<int> chasedData;         // xDot data for the save
	std::vector<Eigen::MatrixXf> WHatData;
	std::vector<Eigen::VectorXf> sigmaData;
	int indexj;
	int indexj1;

public:
	// initialize constructor
	Sheep(std::string sheepNameInit, float K1Init, float kclInit, float GammaInit, float DeltatInit,
	      bool pushToGoalInit, std::vector<float> xgInit, float sm, float s, int L1Init,
	      int NInit, float lambdaInit, bool saveDataInit, std::string runNumberInit, double height) : sheepName(sheepNameInit)
	{
		nh = ros::NodeHandle(sheepName);                       // initialize sheep handle
		xSub = nh.subscribe("mocapPose",1,&Sheep::xCB,this);   // subscribe to sheep pose from mocap
		xDotSub = nh.subscribe("mocapWorldVelFilt",1,&Sheep::xDotCB,this); // subscribe to sheep velocity
		minEigTimer = nh.createTimer(ros::Duration(1.0),&Sheep::minEigCB,this,false);

		// intialize
		x = Eigen::Vector2f::Zero();
		y = Eigen::Vector2f::Zero();
		xg(0) = xgInit.at(0);
		xg(1) = xgInit.at(1);
		pushToGoal = pushToGoalInit;
		firstMocap = true;
		chased = false;
		K1 = K1Init;
		kcl = kclInit;
		Gamma = GammaInit;
		Deltat = DeltatInit;
		unchasedTime = 0.0;
		chasedTime = 0.0;
		killSheep = false;
		dataSaved = false;
		saveData = saveDataInit;
		runNumber = runNumberInit;
		xDotMeas = Eigen::Vector2f::Zero();
		sm2 = std::pow(sm,2);
		s2 = std::pow(s,2);
		L1 = L1Init;
		M = Eigen::MatrixXf::Zero(3,L1);
		yP = Eigen::Vector3f::Zero();
		yQ = Eigen::Vector4f::Zero();
		yQ(0) = 1.0;
		polyOrder = 1;
		firstTime = ros::Time::now();
		N = NInit;
		lambda = lambdaInit;
		Y = Eigen::VectorXf::Zero(L1);
		xData.push_back(Eigen::Vector2f::Zero());
		timeData.push_back(0.0);
		chasedData.push_back(0);
		sigmaData.push_back(Eigen::VectorXf::Zero(L1));


		YTYSumj = Eigen::MatrixXf::Zero(L1,L1);
		YTDxSumj = Eigen::MatrixXf::Zero(L1,2);
		YTYSumj1 = Eigen::MatrixXf::Zero(L1,L1);
		YTDxSumj1 = Eigen::MatrixXf::Zero(L1,2);
		indexj = 0;
		indexj1 = 0;
		minEigj = 0;
		minEigj1 = -1.0;

		std::default_random_engine generator;// generates random numbers
		std::uniform_real_distribution<float> distribution(0.0,sm);//distribution for the random numbers

		//// generate the random basis
		//for (int i = 0; i < L1; i++)
		//{
			//for (int j = 0; j < 2; j++)
			//{
				//M(j,i) = distribution(generator);
				//if ((j == 0) || (j == 1))
				//{
					//M(j,i) = M(j,i);
				//}
				//if (j == 2)
				//{
					//M(j,i) = -1.0*std::fabs(M(j,i)) - 0.05;
				//}
			//}
		//}

		float what = 0.001/((float)L1/4);

		Eigen::MatrixXf MFL = Eigen::MatrixXf::Zero(3,L1/4);
		Eigen::MatrixXf MBL = Eigen::MatrixXf::Zero(3,L1/4);
		Eigen::MatrixXf MBR = Eigen::MatrixXf::Zero(3,L1/4);
		Eigen::MatrixXf MFR = Eigen::MatrixXf::Zero(3,L1/4);

		Eigen::MatrixXf WHatFL = Eigen::MatrixXf::Zero(L1/4,2);
		Eigen::MatrixXf WHatBL = Eigen::MatrixXf::Zero(L1/4,2);
		Eigen::MatrixXf WHatBR = Eigen::MatrixXf::Zero(L1/4,2);
		Eigen::MatrixXf WHatFR = Eigen::MatrixXf::Zero(L1/4,2);

		// generate the random basis
		for (int i = 0; i < L1/4; i++)
		{
			for (int j = 0; j < 2; j++)
			{
				for (int k = 0; k < 3; k++)
				{
					switch (k)
					{
						// front left ++
						case 0:
						{
							MFL(j,i) = distribution(generator);
							WHatFL(i,j) = what;
						}

						// back left -+
						case 1:
						{
							MBL(j,i) = distribution(generator);
							WHatBL(i,j) = what;
							if (j == 0)
							{
								MBL(j,i) = -1.0*MBL(j,i);
								WHatBL(i,j) = -1.0*WHatBL(i,j);
							}
						}

						// back right --
						case 2:
						{
							MBR(j,i) = -1.0*distribution(generator);
							WHatBR(i,j) = -what;
						}

						// front right +-
						case 3:
						{
							MFR(j,i) = distribution(generator);
							WHatFR(i,j) = what;
							if (j == 1)
							{
								MFR(j,i) = -1.0*MFR(j,i);
								WHatFR(i,j) = -1.0*WHatFR(i,j);
							}
						}
					}
				}
			}
			MFL(2,i) = -1.0*height;
			MBL(2,i) = -1.0*height;
			MBR(2,i) = -1.0*height;
			MFR(2,i) = -1.0*height;
		}

		M.block(0,0,3,L1/4) = MFL;
		std::cout << "M:\n" << M << std::endl;
		M.block(0,L1/4,3,L1/4) = MBL;
		std::cout << "M:\n" << M << std::endl;
		M.block(0,2*L1/4,3,L1/4) = MBR;
		std::cout << "M:\n" << M << std::endl;
		M.block(0,3*L1/4,3,L1/4) = MFR;
		std::cout << "M:\n" << M << std::endl;

		Eigen::MatrixXf WHat = Eigen::MatrixXf::Zero(L1,2);
		WHat.block(0,0,L1/4,2) = WHatFL;
		std::cout << "WHat:\n" << WHat << std::endl;
		WHat.block(L1/4,0,L1/4,2) = WHatBL;
		std::cout << "WHat:\n" << WHat << std::endl;
		WHat.block(2*L1/4,0,L1/4,2) = WHatBR;
		std::cout << "WHat:\n" << WHat << std::endl;
		WHat.block(3*L1/4,0,L1/4,2) = WHatFR;
		std::cout << "WHat:\n" << WHat << std::endl;

		WHatData.push_back(WHat);

	}

	// copy constructor
	Sheep(const Sheep& sheepNew) : sheepName(sheepNew.sheepName)
	{
		nh = ros::NodeHandle(sheepName);                         // initialize sheep handle
		xSub = nh.subscribe("mocapPose",1,&Sheep::xCB,this);     // subscribe to sheep pose from mocap
		xDotSub = nh.subscribe("mocapWorldVelFilt",1,&Sheep::xDotCB,this); // subscribe to sheep velocity
		minEigTimer = nh.createTimer(ros::Duration(1.0),&Sheep::minEigCB,this,false);

		// intialize
		xg = sheepNew.xg;
		x = sheepNew.x;
		pushToGoal = sheepNew.pushToGoal;
		firstMocap = true;
		chased = false;
		K1 = sheepNew.K1;
		kcl = sheepNew.kcl;
		Gamma = sheepNew.Gamma;
		Deltat = sheepNew.Deltat;
		unchasedTime = 0.0;
		chasedTime = 0.0;
		killSheep = sheepNew.killSheep;
		dataSaved = sheepNew.dataSaved;
		saveData = sheepNew.saveData;
		runNumber = sheepNew.runNumber;
		xDotMeas = sheepNew.xDotMeas;
		M = sheepNew.M;
	    sm2 = sheepNew.sm2;
		s2 = sheepNew.s2;
		L1 = sheepNew.L1;
		yP = sheepNew.yP;
		yQ = sheepNew.yQ;
		polyOrder = sheepNew.polyOrder;
		firstTime = sheepNew.firstTime;
		N = sheepNew.N;
		lambda = sheepNew.lambda;
		Y = sheepNew.Y;
		YTYSumj = sheepNew.YTYSumj;
		YTDxSumj = sheepNew.YTDxSumj;
		YTYSumj1 = sheepNew.YTYSumj1;
		YTDxSumj1 = sheepNew.YTDxSumj1;
		indexj = sheepNew.indexj;
		indexj1 = sheepNew.indexj1;
		minEigj = sheepNew.minEigj;
		minEigj1 = sheepNew.minEigj1;
		xData = sheepNew.xData;
		timeData = sheepNew.timeData;
		chasedData = sheepNew.chasedData;
		WHatData = sheepNew.WHatData;
		sigmaData = sheepNew.sigmaData;
	}

	// sheep velocity callback
	void xDotCB(const geometry_msgs::TwistStamped::ConstPtr& msg)
	{
		Eigen::Vector2f xDotMeasNew;
		xDotMeasNew << msg->twist.linear.x, msg->twist.linear.y;
		xDotMeas = xDotMeasNew;
	}

	// eigenvalue callback
	void minEigCB(const ros::TimerEvent& event)
	{
		if (minEigj < lambda)
		{
			//get the minimum eigenvalue for the j stack
			Eigen::VectorXf YTYSumjEig = YTYSumj.eigenvalues().real();
			minEigj = YTYSumjEig.minCoeff();
		}
		else
		{
			//get the minimum eigenvalue of the j1 stack
			Eigen::VectorXf YTYSumj1Eig = YTYSumj1.eigenvalues().real();
			minEigj1 = YTYSumj1Eig.minCoeff();
		}
	}

	// get the basis functions
	Eigen::VectorXf getsigma()
	{
		Eigen::VectorXf sigma = Eigen::VectorXf::Zero(L1);
		Eigen::Vector4f y4,x4;
		y4 << 0.0, yP(0), yP(1), yP(2);
		x4 << 0.0, x(0), x(1), 0.0;
		Eigen::Vector4f D = x4 - y4;
		Eigen::Vector4f Dy = getqMat(getqMat(getqInv(yQ))*D)*yQ;
		Eigen::Vector3f X;
		X << Dy(1),Dy(2),Dy(3);
		for (int i = 0; i < L1; i++)
		{
			float XDiff = (X - M.block(0,i,3,1)).transpose()*(X - M.block(0,i,3,1));
			sigma(i) = 1.0/std::sqrt(2.0*M_PIl*s2)*std::exp(-1.0/(2.0*s2)*std::pow(XDiff,2));
		}

		std::cout << sheepName << " sigma norm " << sigma.norm() << std::endl;

		return sigma;
	}

	// get the x difference approx
    Eigen::RowVector2f getDx()
    {
		int bufferSize = xBuffer.size();
		Eigen::MatrixXf X = Eigen::MatrixXf::Zero(2*bufferSize,1);
		Eigen::MatrixXf H = Eigen::MatrixXf::Zero(2*bufferSize,4);
		Eigen::MatrixXf II = Eigen::MatrixXf::Identity(2,2);
		for (int i = 0; i < bufferSize; i++)
		{
			float timei = (timeBuffer.at(i) - timeBuffer.at(0)).toSec();
			int start = 2*i;
			X.block(start,0,2,1) = xBuffer.at(i);
			Eigen::MatrixXf Hi(2,4);
			Hi << Eigen::MatrixXf::Identity(2,2), timei*II;
			H.block(start,0,2,4) = Hi;
		}
		Eigen::Matrix4f HTH = H.transpose()*H;
		Eigen::Vector4f HTX = H.transpose()*X;
		Eigen::Vector4f theta = HTH.ldlt().solve(HTX);
		float Dt = (timeBuffer.at(bufferSize-1) - timeBuffer.at(0)).toSec();
		Eigen::RowVector2f XD = Dt*theta.segment(2,2).transpose();

		std::cout << sheepName << " XD norm " << XD.norm() << std::endl;

		return XD;
	}

	// get integral of Y approx
	Eigen::RowVectorXf getscriptY()
	{
		//int bufferSize = YBuffer.size();
		//Eigen::MatrixXf X = Eigen::MatrixXf::Zero(L1*bufferSize,1);
		//Eigen::MatrixXf H = Eigen::MatrixXf::Zero(L1*bufferSize,L1*(polyOrder+1));
		//Eigen::MatrixXf II = Eigen::MatrixXf::Identity(L1,L1);
		//for (int i = 0; i < bufferSize; i++)
		//{
			//float timei = (timeBuffer.at(i) - timeBuffer.at(0)).toSec();
			//int start = L1*i;
			//X.block(start,0,L1,1) = YBuffer.at(i);
			//for (int j = 0; j <= polyOrder; j++)
			//{
				//int startj = L1*j;
				//H.block(start,startj,L1,L1) = std::pow(timei,j)*II;
			//}
		//}
		////std::cout << "H\n" << H << std::endl;
		//Eigen::MatrixXf HTH = H.transpose()*H;
		////std::cout << "HTH\n" << HTH << std::endl;
		//Eigen::VectorXf HTX = H.transpose()*X;
		//Eigen::VectorXf theta = HTH.ldlt().solve(HTX);
		////std::cout << "HTX\n" << HTX << std::endl;
		////std::cout << "theta\n" << theta << std::endl;
		//float Dt = (timeBuffer.at(bufferSize-1) - timeBuffer.at(0)).toSec();
		//Eigen::RowVectorXf integratedBuffer = theta.segment(0,L1).transpose();
		//for (int i = 1; i <= polyOrder; i++)
		//{
			//int start = L1*i;
			//integratedBuffer = integratedBuffer + (std::pow(Dt,i)/((float)i))*theta.segment(start,L1).transpose();

		//}

		Eigen::RowVectorXf integratedBuffer = Eigen::RowVectorXf::Zero(L1);
		for (int i = 0; i < timeBuffer.size()-1; i++)
		{
			integratedBuffer += (0.5*(timeBuffer.at(i+1) - timeBuffer.at(i)).toSec()*(YBuffer.at(i+1) + YBuffer.at(i))).transpose();
		}

		//std::cout << "integratedBuffer\n" << integratedBuffer << std::endl;

		std::cout << sheepName << " integratedBuffer norm " << integratedBuffer.norm() << std::endl;

		return integratedBuffer;
	}

	//update a stack if the new value increases the eigenvalue
	bool updateStack(Eigen::MatrixXf& YTYSum, Eigen::MatrixXf& YTDxSum, Eigen::RowVectorXf scriptY, Eigen::RowVector2f Dx, int& index)
	{
		Eigen::Vector4f Dx4(0.0,Dx(0),Dx(1),0.0);
		Dx4 = getqMat(getqMat(getqInv(yQ))*Dx4)*yQ;
		Eigen::RowVector2f DxQ(Dx4(1),Dx4(2));
		Eigen::MatrixXf YTY = scriptY.transpose()*scriptY;
		Eigen::MatrixXf YTDx = scriptY.transpose()*DxQ;

		std::cout << sheepName << " DxQ norm " << DxQ.norm() << std::endl;

		if ((0.1 < DxQ.norm()) && (DxQ.norm() < 0.333))
		{
			YTYSum += YTY;
			YTDxSum += YTDx;
			index += 1;
		}

		//if (index < N)
		//{
			//YTYSum += YTY;
			//YTDxSum += YTDx;
			//index += 1;
		//}
		//else
		//{
			//// get the minimum eigenvalue of the stack
			//Eigen::VectorXf YTYSumEig = YTYSum.eigenvalues().real();
			//float minEig = YTYSumEig.minCoeff();

			//// offset the initial sum by the new measure
			//Eigen::MatrixXf YTYSumExt = YTYSum + YTY;

			//// get the minimum eigenvalue of the extended stack
			//Eigen::VectorXf YTYSumExtEig = YTYSumExt.eigenvalues().real();
			//float minEigExt = YTYSumExtEig.minCoeff();

			////if the extended stack increases the minimum eigenvalue then use the data
			//if (minEigExt > minEig)
			//{
				//YTYSum += YTY;
				//YTDxSum += YTDx;
				//index += 1;
			//}
		//}
	}

	//WHatDot
	Eigen::MatrixXf getWHatDot(Eigen::MatrixXf WHat, Eigen::VectorXf sigma, Eigen::Vector2f ey, Eigen::MatrixXf YTYSum, Eigen::MatrixXf YTDxSum)
	{
		Eigen::MatrixXf WHatDot = kcl*Gamma*(YTDxSum - YTYSum*WHat);
		//Eigen::MatrixXf WHatDot = Eigen::MatrixXf::Zero(L1,2);
		if (chased)
		{
			WHatDot = WHatDot + Gamma*K1*sigma*ey.transpose();
		}

		//std::cout << sheepName << " whatdot " << std::endl;

		return WHatDot;
	}

	//integrate WHat
	Eigen::MatrixXf integrateWHat(Eigen::MatrixXf WHat, Eigen::VectorXf sigma, Eigen::Vector2f ey, Eigen::MatrixXf YTYSum, Eigen::MatrixXf YTDxSum, float dt)
	{
		Eigen::MatrixXf C1 = getWHatDot(WHat, sigma, ey, YTYSum, YTDxSum);
		Eigen::MatrixXf C2 = getWHatDot(WHat+dt*C1/2.0, sigma, ey, YTYSum, YTDxSum);
		Eigen::MatrixXf C3 = getWHatDot(WHat+dt*C2/2.0, sigma, ey, YTYSum, YTDxSum);
		Eigen::MatrixXf C4 = getWHatDot(WHat+dt*C3, sigma, ey, YTYSum, YTDxSum);
		Eigen::MatrixXf WHatDot = (C1 + 2.0*C2 + 2.0*C3 + C4)/6.0;

		std::cout << sheepName << " integrate what " << std::endl;

		return WHat + dt*WHatDot;
	}

	// sheep pose callback
	void xCB(const geometry_msgs::PoseStamped::ConstPtr& msg)
	{
		if (dataSaved)
		{
			std::cout << sheepName << " surrendered\n";
			xSub.shutdown();
			xDotSub.shutdown();
			minEigTimer.stop();
			return;
		}

		ros::Time timeNew = msg->header.stamp;
		Eigen::Vector2f xNew;
		xNew << msg->pose.position.x, msg->pose.position.y;
		x = xNew;

		//Eigen::Vector4f y4,x4;
		//y4 << 0.0, yP(0), yP(1), yP(2);
		//x4 << 0.0, x(0), x(1), 0.0;
		//Eigen::Vector4f D = x4 - y4;
		////Eigen::Vector4f D = x4;
		//D = getqMat(getqMat(getqInv(yQ))*D)*yQ;
		//Eigen::Vector2f xQ(D(1),D(2));

		if (firstMocap)
		{
			firstMocap = false;
			timeLast = timeNew;
			firstTime = timeNew;
		}
		Eigen::VectorXf sigma = getsigma();

		timeBuffer.push_back(timeNew);
		xBuffer.push_back(x);
		YBuffer.push_back(sigma);
		float dt = (timeNew - timeLast).toSec();
		timeLast = timeNew;

		if (timeBuffer.size() >= 3)
		{
			// while the buffer is too big pop off the oldest data as long as it wont make
			// the time on the buffer too small. compare with the second oldest data to ensure
			// the buffer stays large enough
			while ((timeBuffer.at(timeBuffer.size()-1) - timeBuffer.at(1)).toSec() > Deltat)
			{
				timeBuffer.pop_front();
				xBuffer.pop_front();
				YBuffer.pop_front();
			}

			// if enough time has passed begin updating stacks
			if (((timeNew - firstTime).toSec() >= Deltat) && chased)
			{
				Eigen::RowVector2f Dx = getDx();

				Eigen::RowVectorXf scriptY = getscriptY();

				//std::cout << "time3 " << (loopNow - loopStart).toSec() << std::endl;
				//std::cout << "Dx\n" << Dx << std::endl;
				//std::cout << "M\n" << M << std::endl;
				//std::cout << "scriptY\n" << scriptY << std::endl;
				//std::cout << minEigj << std::endl;
				// if enough data on the stack check to see if it is positive

				// if the minimum eigenvalue is large enough then stop adding to the j stack and start adding to the j1 stack
				//if (minEigj > lambda)
				//{
					//updateStack(YTYSumj1,YTDxSumj1,scriptY,Dx,indexj1);
					//// if the minimum eigenvalue of the j1 stack is large enough then switch the stacks and clear the j1 stack
					//if (minEigj1 > lambda)
					//{
						//YTYSumj = YTYSumj1;
						//YTDxSumj = YTDxSumj1;
						//indexj = indexj1;

						//YTYSumj1 = Eigen::MatrixXf::Zero(L1,L1);
						//YTDxSumj1 = Eigen::MatrixXf::Zero(L1,2);
						//indexj1 = 0;
					//}
				//}
				//else
				//{
					//updateStack(YTYSumj,YTDxSumj,scriptY,Dx,indexj);
				//}
				updateStack(YTYSumj,YTDxSumj,scriptY,Dx,indexj);
			}
		}

		// update the weight estimates
		Eigen::Vector2f xBar = x - xg;
		Eigen::Vector2f yd = K1*xBar + xg;
		Eigen::Vector2f ey = yd - y;

		Eigen::Vector4f ey4(0.0, ey(0), ey(1), 0.0);
		ey4 = getqMat(getqMat(getqInv(yQ))*ey4)*yQ;
		Eigen::Vector2f eyQ = Eigen::Vector2f(ey4(1),ey4(2));

		//std::cout << "eyQ\n" << eyQ << std::endl;

		Eigen::MatrixXf WHat = integrateWHat(WHatData.at(WHatData.size()-1),sigma,eyQ,YTYSumj,YTDxSumj,dt);

		//std::cout << sheepName << "\n" << eyQ << std::endl;

		// if the kill sheep has not been signaled then keep saving data otherwise save the data if it hasnt been saved yet
		if (!killSheep)
		{
			WHatData.push_back(WHat);
			timeData.push_back((timeNew - firstTime).toSec());
			xData.push_back(x);
			sigmaData.push_back(sigma);

			int amIChased = 0;
			if (chased)
			{
				amIChased = 1;
			}
			chasedData.push_back(amIChased);

			//std::cout << "M\n" << M << std::endl;
			//std::cout << "WHatT\n" << WHat.transpose() << std::endl;
			//std::cout << sheepName << " time " << (timeNew - firstTime).toSec() << std::endl;
		}
		else
		{
			if (!dataSaved && saveData)
			{
				saveRunData();
			}
			dataSaved = true;
		}
	}

	// save the data
	void saveRunData()
	{
		std::ofstream sheepFile("/home/ncr/ncr_ws/src/single_agent_herding_n_agents_nn/experiments/"+sheepName+runNumber+".txt");
		if (sheepFile.is_open())
		{
			sheepFile << "time," << "x0," << "x1," << "chased,";
			for (int i = 0; i < L1; i++)
			{
				sheepFile << "WHat" << i << "0," << "WHat" << i << "1,";
			}
			for (int i = 0; i < L1; i++)
			{
				sheepFile << "sigma" << i << ",";
			}
			for (int i = 0; i < L1; i++)
			{
				sheepFile << "M" << i << "0," << "M" << i << "1,";
			}
			sheepFile << "\n";

			for (int i = 0; i < xData.size(); i++)
			{
				float timei = timeData.at(i);
				Eigen::Vector2f xi = xData.at(i);
				float x0i = xi(0);
				float x1i = xi(1);
				int chasedi = chasedData.at(i);
				sheepFile << timei << "," << x0i << "," << x1i << "," << chasedi << ",";

				Eigen::MatrixXf WHati = WHatData.at(i);
				Eigen::MatrixXf sigmai = sigmaData.at(i);
				for (int j = 0; j < L1; j++)
				{
					Eigen::RowVector2f WHatij = WHati.block(j,0,1,2);
					sheepFile << WHatij(0) << "," << WHatij(1) << ",";
				}
				for (int j = 0; j < L1; j++)
				{
					sheepFile << sigmai(j) << ",";
				}
				for (int j = 0; j < L1; j++)
				{
					Eigen::Vector3f Mj = M.block(0,j,3,1);
					sheepFile << Mj(0) << "," << Mj(1) << ",";
				}
				sheepFile << "\n";
			}
			sheepFile.close();
		}
	}

	//get the error
	Eigen::Vector2f geteyQ()
	{
		// update the weight estimates
		Eigen::Vector2f xBar = x - xg;
		Eigen::Vector2f yd = K1*xBar + xg;
		Eigen::Vector2f ey = yd - y;

		Eigen::Vector4f ey4;
		ey4 << 0.0, ey(0), ey(1), 0.0;
		ey4 = getqMat(getqMat(yQ)*ey4)*getqInv(yQ);

		Eigen::Vector2f eyQ(ey4(1),ey4(2));
		return eyQ;
	}

	// return sheep position and if this is the first call since the sheep
	// started being chased
	Eigen::Vector2f getx()
	{
		return x;
	}

	// return sheep xBar
	Eigen::Vector2f getxg()
	{
		return xg;
	}

	// return sheep xBar
	Eigen::Vector2f getxBar()
	{
		return x-xg;
	}

	// return sheep first mocap
	bool getfirstMocap()
	{
		return firstMocap;
	}

	// bear position update
	void sety(Eigen::Vector2f yNew)
	{
		y = yNew;
	}

	// bear 3d pose update
	void setypose(Eigen::Vector3f yPNew, Eigen::Vector4f yQNew)
	{
		yP = yPNew;
		yQ = yQNew;
	}

	// chased update
	void setchased(bool chasedNew)
	{
		chased = chasedNew;
	}

	//get WHat
	Eigen::MatrixXf getWHat()
	{
		return WHatData.at(WHatData.size()-1);
	}

	void setkillSheep()
	{
		killSheep = true;
	}

	bool getdataSaved()
	{
		return dataSaved;
	}

};

// main algorithm from paper
// Single Agent Herding of n-Agents: A Switched Systems Approach
// 1st author Ryan A. Licitra
// notation should be the same as in the paper
class Bear
{
	const std::string bearName;    // bear bebop name
	ros::NodeHandle nh;            // nodehandle
	ros::Subscriber ySub;          // bear pose subscriber, subscribe to pose message
	ros::Publisher yPub;           // bear position publisher
	ros::Publisher uyPub;          // bear linear velocity command publisher, publish odom message
	ros::Subscriber yDotSub;       // bear velocity subscriber, subscribe to twist message
	bool pushToGoal;			   // boolean to indicate the sheep should be pushed to goal
	bool firstMocap;               // indicates first mocap for bear has been received
	std::vector<Sheep> sheep;      // sheep
	float k1, k2, k3, ky, ks, K1, K2;         // controller gains
	float kcl;                    // constant learning gain
	float Deltat;                 // integration window
	int N;                         // history stack size
	bool useRobust;
	Eigen::Vector2f y;             // bear position
	Eigen::Vector3f yP;            // bear position 3D
	Eigen::Vector4f yQ;            // bear orientation in 3D
	Eigen::Vector4f yQLast;            // bear orientation in 3D
	Eigen::Vector2f yDotMeas;      // bear measured velocity
	Eigen::Vector3f yF;            // final bear position
	Eigen::Vector4f yQF;            // final bear orientation
	float Gamma;         // constant estimator/learning gain
	int chasing;                   // indicates which sheep is currently being chased
	float herdingRatio;           // ratio for how much the bear should drive down the sheep before switching
	float xNormChasingStart;      // sheep norm at the begining of being chased
	ros::Time timeLast;            // last time
	bool sheepFound;               // indicate that all the sheep have been seen by mocap
	float height;                 // height for the bear to hover at
	float originRadius;           // radius to drive sheep within around the origin
	Eigen::Vector4f wall;          // wall for the bear
	bool killBear;                 // kill the bear
	std::vector<Eigen::Vector2f> yData;  // y data for the save
	std::vector<Eigen::Vector2f> yDotData;  // y data for the save
	std::vector<Eigen::Vector2f> uyData; // yDot data for the save
	std::vector<Eigen::Vector2f> uy1Data; // yDot data for the save
	std::vector<Eigen::Vector2f> uy2Data; // yDot data for the save
	std::vector<Eigen::Vector2f> uy3Data; // yDot data for the save
	std::vector<Eigen::Vector2f> eyData; // yDot data for the save
	std::vector<float> timeData;     // time data for the save
	std::vector<int> chasingData;
	std::string runNumber;         // run number for the save
	float velMaxRelDiff;       // maximum relative difference between commanded velocity and measured velocity
	bool saveData;                 // indicator for saving the experiment
	bool dataSaved;                // indicator to show the data has been saved
	float sm2;                          // variance to choose the means
	float s2;							 // variance for the weights
	int L1;                              // number of neurons
	float lambda;
	ros::Time firstTime;

public:

	// initialize constructor
	Bear(std::string runNumberInit, std::string bearNameInit, std::vector<std::string> sheepNameInit,
	     float k1Init, float k2Init, float k3Init, float ksInit, float kyInit, float kclInit,
	     float gammaInit, float DeltatInit, int NInit,
	     float herdingRatioInit, float heightInit,
	     float originRadiusInit, std::vector<float> bearWall, bool pushToGoalInit,
	     std::vector<float> xgInit, float velMaxRelDiffInit, bool saveDataInit,
	     float sm, float s, int L1Init, float lambdaInit, bool useRobustInit) : bearName(bearNameInit)
	{
		runNumber = runNumberInit;
		nh = ros::NodeHandle(bearName);                        // initialize bear handle
		ySub = nh.subscribe("mocapPose",1,&Bear::yCB,this);    // initialize bear pose from mocap
		uyPub = nh.advertise<nav_msgs::Odometry>("desOdom",1); // initialize bear linear velocity command from algorithm
		yDotSub = nh.subscribe("mocapWorldVelFilt",1,&Bear::yDotCB,this);

		// initialize
		useRobust = useRobustInit;
		firstMocap = true;
		pushToGoal = pushToGoalInit;
		k1 = k1Init;
		k2 = k2Init;
		ky = kyInit;
		k3 = k3Init;
		ks = ksInit;
		K1 = k1 + k2;
		K2 = ky + k3;
		ky = kyInit;
		kcl = kclInit;
		Gamma = gammaInit;
		Deltat = DeltatInit;
		N = NInit;
		chasing = -1;
		sheepFound = false;
		height = heightInit;
		herdingRatio = herdingRatioInit;
		originRadius = originRadiusInit;
		wall(0) = bearWall.at(0);
		wall(1) = bearWall.at(1);
		wall(2) = bearWall.at(2);
		wall(3) = bearWall.at(3);
		velMaxRelDiff = velMaxRelDiffInit;
		yDotMeas = Eigen::Vector2f::Zero();
		//yDotLast = Eigen::Vector2f::Zero();
		saveData = saveDataInit;
		dataSaved = false;
		sm2 = std::pow(sm,2);
		s2 = std::pow(s,2);
		L1 = L1Init;
		lambda = lambdaInit;
		yF = Eigen::Vector3f(0.0,0.0,1.5);
		yQF = Eigen::Vector4f(1.0,0.0,0.0,0.0);
		yQLast = Eigen::Vector4f(1.0,0.0,0.0,0.0);

		// initialize the sheep
		for (int i = 0; i < sheepNameInit.size(); i++)
		{
			int xgi1Index = 2*i;
			int xgi2Index = 2*i+1;
			std::vector<float> xgi = {xgInit.at(xgi1Index),xgInit.at(xgi2Index)};
			sheep.push_back(Sheep(sheepNameInit.at(i),K1,kcl,Gamma,Deltat,pushToGoal,xgi,sm,s,L1,N,lambda,saveData,runNumber,height));
		}

		killBear = false;
	}

	// bear velocity callback
	void yDotCB(const geometry_msgs::TwistStamped::ConstPtr& msg)
	{
		Eigen::Vector2f yDotMeasNew;
		yDotMeasNew << msg->twist.linear.x, msg->twist.linear.y;
		yDotMeas = yDotMeasNew;
	}

	// bear pose callback
	void yCB(const geometry_msgs::PoseStamped::ConstPtr& msg)
	{
		if (dataSaved)
		{
			nav_msgs::Odometry uyMsg;
			uyMsg.pose.pose.position.x = yF(0);
			uyMsg.pose.pose.position.y = yF(1);
			uyMsg.pose.pose.position.z = yF(2);
			uyMsg.pose.pose.orientation.w = yQF(0);
			uyMsg.pose.pose.orientation.x = yQF(1);
			uyMsg.pose.pose.orientation.y = yQF(2);
			uyMsg.pose.pose.orientation.z = yQF(3);
			uyPub.publish(uyMsg);

			//check if the sheep have saved the data/finished cleanly and if they have shutdown
			bool sheepSavedData = true;
			std::vector<int> sheepToSurrender;
			for (int i = 0; i < sheep.size(); i++)
			{
				sheepSavedData &= sheep.at(i).getdataSaved();
				if (!sheep.at(i).getdataSaved())
				{
					sheepToSurrender.push_back(i);
				}
			}

			if (sheepSavedData)
			{
				std::cout << bearName << " is victorious\n";
				ros::shutdown();
				ySub.shutdown();
				yDotSub.shutdown();
			}
			else
			{
				std::cout << "waiting for";
				for (int i = 0; i < sheepToSurrender.size(); i++)
				{
					std::cout << " sheep " << sheepToSurrender.at(i);
				}
				std::cout << " to surrender\n";
			}
			return;
		}

		//std::cout << "hi1\n";
		//std::cout << bearName << std::endl;
		// bear new position
		Eigen::Vector2f yNew;
		yNew << msg->pose.position.x, msg->pose.position.y;
		y = yNew;

		Eigen::Vector3f yPNew;
		yPNew << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
		yP = yPNew;

		Eigen::Vector4f yQNew;
		yQNew << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;
		yQ = yQNew;

		ros::Time timeNew = msg->header.stamp;

		// if this is the first mocap indicate and start time
		if (firstMocap)
		{
			firstMocap = false;
			timeLast = timeNew;
			firstTime = timeNew;
			yQLast = yQ;
		}

		Eigen::Vector4f yQN = -1.0*yQ;
		if ((yQLast - yQN).norm() < (yQLast - yQ).norm())
		{
			yQ = yQN;
		}
		yQLast = yQ;

		float dt = (timeNew - timeLast).toSec();
		timeLast = timeNew;

		// update bear position for each sheep
		//std::vector<Eigen::Vector2f> xi; // sheep positions
		for (int i = 0; i < sheep.size(); i++)
		{
			sheep.at(i).sety(y);
			sheep.at(i).setypose(yP,yQ);
		}

		// check if sheep found, if they are not then return
		if (!sheepFound)
		{
			for (int i = 0; i < sheep.size(); i++)
			{
				bool sheepFoundi = !sheep.at(i).getfirstMocap();
				if (!sheepFoundi)
				{
					return;
				}
			}
			sheepFound = true;
		}

		// get sheep positions, pick a new target to chase if the chasing index is less than 0
		// chase the target that is the farthest away from the goal in the xy
		if (chasing < 0)
		{
			//std::cout << "new chasing starting norm is " << xNormChasingStart << std::endl;
			for (int i = 0; i < sheep.size(); i++)
			{
				sheep.at(i).setchased(false);
			}

			//std::cout << "entered chase set, old chasing is " << chasing << std::endl;
			std::vector<float> rcColas; // L2 norm of each sheep position in xy
			std::vector<float> xiNorms; // L2 norm of each sheep position in xy
			std::vector<int> xIndexs;
			for (int i = 0; i < sheep.size(); i++)
			{
				Eigen::Vector2f xi = sheep.at(i).getx();
				Eigen::Vector2f xgi = sheep.at(i).getxg();
				Eigen::Vector2f xBari = xi - xgi;
				Eigen::Vector2f yBari = xi - y;
				float xiNorm = 0.0;
				float rcColai = 0.0;
				if (pushToGoal)
				{
					rcColai = 1.25*xBari.norm()-yBari.norm();
					xiNorm = xBari.norm();
				}
				else
				{
					rcColai = 1.25*xi.norm()-yBari.norm();
					xiNorm = xi.norm();
				}

				std::cout << "xiNorm i " << i << " " << xiNorm << std::endl;
				std::cout << "rcColai i " << i << " " << rcColai << std::endl;

				if (xiNorm > originRadius)
				{
					xIndexs.push_back(i);
					rcColas.push_back(rcColai);
					xiNorms.push_back(xiNorm);
					//std::cout << "xiNorm for sheep " << i << " is " << xiNorm << std::endl;
				}
			}

			if (rcColas.size() > 0)
			{
				int chasingIndex = std::distance(rcColas.begin(),std::max_element(rcColas.begin(),rcColas.end())); // chase index of maximum norm
				std::cout << "chasingIndex " << chasingIndex << std::endl;
				std::cout << "rcColas.size() " << rcColas.size() << std::endl;
				chasing = xIndexs.at(chasingIndex);
				xNormChasingStart = xiNorms.at(chasingIndex);

			}
			else
			{
				if (!killBear && !dataSaved)// if all the sheep are within the radius, then save, send out zeros, and shutdown the node
				{
					shutdownNode();
					dataSaved = true;
					return;
				}
			}

			for (int i = 0; i < sheep.size(); i++)
			{
				if (i==chasing)
				{
					sheep.at(i).setchased(true);
				}
			}

			std::cout << "new chasing is " << chasing << std::endl;
		}


		//std::cout << "hi1\n";
		Eigen::Vector2f ey = Eigen::Vector2f::Zero();
		Eigen::VectorXf sigma = Eigen::VectorXf::Zero(L1);
		Eigen::MatrixXf WHat = Eigen::MatrixXf::Zero(L1,2);
		Eigen::Vector2f xBar = Eigen::Vector2f::Zero();
		Eigen::Vector2f x = Eigen::Vector2f::Zero();
		Eigen::Vector2f xHatDot = Eigen::Vector2f::Zero();
		Eigen::Vector2f xg = Eigen::Vector2f::Zero();
		Eigen::Vector4f yQD = yQ;

		//std::cout << "hi2\n";
		if (chasing >= 0)
		{
			//std::cout << "chasing bebop" << chasing+2 << std::endl;
			// get currently chased sheep parameters
			x = sheep.at(chasing).getx();   // most recent sheep position
			xg = sheep.at(chasing).getxg();             // goal for the sheep

			//std::cout << "hi3\n";

			// update the weight estimates
			xBar = x - xg;
			Eigen::Vector2f yd = K1*xBar + xg;
			ey = yd - y;

			sigma = sheep.at(chasing).getsigma();
			//std::cout << "hi31\n";

			WHat = sheep.at(chasing).getWHat();

			//std::cout << "hi4\n";

			xHatDot = WHat.transpose()*sigma;
			Eigen::Vector4f xHatDot4;
			xHatDot4 << 0.0, xHatDot(0), xHatDot(1), 0.0;
			xHatDot4 = getqMat(getqMat(yQ)*xHatDot4)*getqInv(yQ);
			xHatDot = Eigen::Vector2f(xHatDot4(1),xHatDot4(2));

			float headingd = std::atan2(xg(1)-y(1),xg(0)-y(0));
			yQD = Eigen::Vector4f(std::cos(0.5*headingd), 0.0, 0.0, std::sin(0.5*headingd));
		}

		Eigen::Vector2f eysign(std::copysign(1.0,ey(0)),std::copysign(1.0,ey(1)));
		Eigen::Vector2f uy1 = K2*ey;
		Eigen::Vector2f uy2 = K1*xHatDot;
		Eigen::Vector2f uy3 = ks*eysign;
		//Eigen::Vector2f uy = uy1 + uy2 + uy3; // update command
		Eigen::Vector2f uy = Eigen::Vector2f::Zero();

		if (useRobust)
		{
			uy2 = Eigen::Vector2f::Zero();
			uy3 *= (x-y).norm();
			uy = uy1 + uy3; // update command;
		}
		else
		{
			uy = uy1 + uy2 + uy3; // update command;
		}

		//Eigen::Vector2f uy = K2*ey; // update command
		//yDotLast = uy;

		Eigen::Vector4f yQDN = -1.0*yQD;
		if ((yQ - yQDN).norm() < (yQ - yQD).norm())
		{
			yQD = yQDN;
		}

		Eigen::Vector4f qhe = getqMat(getqInv(yQD))*yQ;

		//std::cout << "ey\n" << ey << std::endl << std::endl;
		//std::cout << "xBar\n" << xBar << std::endl << std::endl;
		//std::cout << "K2\n" << K2 << std::endl << std::endl;
		std::cout << "K2*ey\n" << uy1 << std::endl << std::endl;
		std::cout << "K1*xHatDot\n" << uy2 << std::endl << std::endl;
		std::cout << "ks*eysign\n" << uy3 << std::endl << std::endl;
		//std::cout << "chasing " << chasing << std::endl << std::endl;
		//std::cout << "xBar\n" << xBar << std::endl << std::endl;
		//std::cout << "yd\n" << K1*xBar + xg << std::endl << std::endl;
		//std::cout << "sigma\n" << sigma << std::endl << std::endl;
		//std::cout << "WHat\n" << WHat << std::endl << std::endl;
		//std::cout << "WHat^T*sigma\n" << WHat.transpose()*sigma << std::endl << std::endl;
		//std::cout << "xHatDot\n" << xHatDot << std::endl << std::endl;
		//std::cout << "ks*eysign\n" << ks*eysign << std::endl << std::endl;
		//std::cout << "y\n" << y << std::endl;
		//std::cout << "heading\n" << heading << std::endl;
		//std::cout << "headingd\n" << headingd << std::endl;

		//std::cout << "hi5\n";

		// check if the chased sheep has gone under the desired ratio,
		// if it has then reset the chasing indicator
		float xNorm = originRadius;
		if (pushToGoal)
		{
			xNorm = xBar.norm();
		}
		else
		{
			xNorm = x.norm();
		}

		//std::cout << "x for sheep " << chasing << " is " << x << std::endl;
		//std::cout << "xNorm for sheep " << chasing << " is " << xNorm << std::endl;
		if ((xNorm <= xNormChasingStart*herdingRatio) || (xNorm <= originRadius))
		{
			//std::cout << "restting chase" << std::endl;
			chasing = -1;
		}

		//Eigen::Vector4f qx(std::cos(heading),0.0,0.0,std::sin(heading));
		//Eigen::Vector4f quy = getqMat(yQ)*qx;

		//Eigen::Vector4f uyQ(0.0,uy(0),uy(1),0.0);
		//uyQ = getqMat(getqMat(yQ)*uyQ)*getqInv(yQ);

		// send out bear command
		nav_msgs::Odometry uyMsg;

		if (killBear)
		{
			uyMsg.pose.pose.position.x = yF(0);
			uyMsg.pose.pose.position.y = yF(1);
			uyMsg.pose.pose.position.z = yF(2);
			uyMsg.pose.pose.orientation.w = yQF(0);
			uyMsg.pose.pose.orientation.x = yQF(1);
			uyMsg.pose.pose.orientation.y = yQF(2);
			uyMsg.pose.pose.orientation.z = yQF(3);
		}
		else
		{
			Eigen::Vector2f yd = y;
			Eigen::Vector2f yDotd = uy;
			wallCheck(yd,yDotd,wall,dt);
			uyMsg.pose.pose.position.x = yd(0);
			uyMsg.pose.pose.position.y = yd(1);
			uyMsg.pose.pose.position.z = height;
			uyMsg.pose.pose.orientation.w = yQ(0);
			uyMsg.pose.pose.orientation.x = yQ(1);
			uyMsg.pose.pose.orientation.y = yQ(2);
			uyMsg.pose.pose.orientation.z = yQ(3);

			uyMsg.twist.twist.linear.x = yDotd(0);
			uyMsg.twist.twist.linear.y = yDotd(1);
			uyMsg.twist.twist.angular.z = -2.0*qhe(3);
		}
		uyPub.publish(uyMsg);

		if (!dataSaved)
		{
			yData.push_back(y);
			yDotData.push_back(yDotMeas);  // y data for the save
			uyData.push_back(uy);
			uy1Data.push_back(uy1);
			uy2Data.push_back(uy2);
			uy3Data.push_back(uy3);
			timeData.push_back((timeNew - firstTime).toSec());
			chasingData.push_back(chasing);
			eyData.push_back(ey);

			std::cout << bearName << " time " << (timeNew - firstTime).toSec() << std::endl;
			std::cout << std::endl << std::endl << "chasing " << chasing << std::endl << std::endl;
		}
	}

	// sends a kill command to all the sheep and bear and saves the data and kills the node
	void shutdownNode()
	{
		killBear = true;
		yF(0) = y(0);
		yF(1) = y(1);
		yQF = yQ;

		nav_msgs::Odometry uyMsg;
		uyMsg.pose.pose.position.x = yF(0);
		uyMsg.pose.pose.position.y = yF(1);
		uyMsg.pose.pose.position.z = yF(2);
		uyMsg.pose.pose.orientation.w = yQF(0);
		uyMsg.pose.pose.orientation.x = yQF(1);
		uyMsg.pose.pose.orientation.y = yQF(2);
		uyMsg.pose.pose.orientation.z = yQF(3);
		uyPub.publish(uyMsg);

		for (int i = 0; i < sheep.size(); i++)
		{
			sheep.at(i).setkillSheep();
		}
		if (saveData && !dataSaved)
		{
			std::ofstream bearFile("/home/ncr/ncr_ws/src/single_agent_herding_n_agents_nn/experiments/"+bearName+runNumber+".txt");
			if (bearFile.is_open())
			{
				bearFile << "time," << "y0," << "y1," << "yDot0," << "yDot1," << "uy0," << "uy1," << "uy10," << "uy11," << "uy20," << "uy21," << "uy30," << "uy31," << "chasing," << "ey0," << "ey1" << "\n";
				for (int i = 0; i < yData.size(); i++)
				{
					float timei = timeData.at(i);
					Eigen::Vector2f yi = yData.at(i);
					float y0i = yi(0);
					float y1i = yi(1);
					Eigen::Vector2f yDoti = yDotData.at(i);
					float yDot0i = yDoti(0);
					float yDot1i = yDoti(1);
					Eigen::Vector2f uyi = uyData.at(i);
					float uy0i = uyi(0);
					float uy1i = uyi(1);
					Eigen::Vector2f uy_1i = uy1Data.at(i);
					float uy10i = uy_1i(0);
					float uy11i = uy_1i(1);
					Eigen::Vector2f uy_2i = uy2Data.at(i);
					float uy20i = uy_2i(0);
					float uy21i = uy_2i(1);
					Eigen::Vector2f uy_3i = uy3Data.at(i);
					float uy30i = uy_3i(0);
					float uy31i = uy_3i(1);
					int chasingi = chasingData.at(i);
					Eigen::Vector2f eyi = eyData.at(i);
					float ey0i = eyi(0);
					float ey1i = eyi(1);
					bearFile << timei << "," << y0i << "," << y1i << "," << yDot0i << "," << yDot1i << "," << uy0i << "," << uy1i << "," << uy10i << "," << uy11i << "," << uy20i << "," << uy21i << "," << uy30i << "," << uy31i << "," << chasingi << "," << ey0i << "," << ey1i << "\n";
				}
				bearFile.close();
			}
		}
	}
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "herding");

	//handle to launch file parameters
	ros::NodeHandle nhp("~");

	std::string runNumber;
	bool saveData;
	nhp.param<std::string>("runNumber",runNumber,"x");
	nhp.param<bool>("saveData",saveData,false);

	// bear bebop name
	std::string bearName;
	nhp.param<std::string>("bearName",bearName,"bebop");

	// sheep bebop name
	std::vector<std::string> sheepName;
	nhp.param<std::vector<std::string>>("sheepName",sheepName,{"sheep"});

	// constant gains and the bound ratio
	float k1, k2, k3, ks, ky, kcl, height, Gamma, Deltat, herdingRatio, originRadius, velMaxRelDiff, sm, s, lambda;
	int N, L1quad;
	bool pushToGoal,useRobust;
	std::vector<float> heights,bearWall,sheepWall,xg;
	nhp.param<float>("k1",k1,1.0);
	nhp.param<float>("k2",k2,2.55);
	nhp.param<float>("k3",k3,2.55);
	nhp.param<float>("ks",ks,2.55);
	nhp.param<float>("ky",ky,1.0);
	nhp.param<float>("kcl",kcl,1.0);
	nhp.param<float>("Gamma",Gamma,0.1);
	nhp.param<float>("Deltat",Deltat,0.2);
	nhp.param<int>("N",N,10);
	nhp.param<float>("herdingRatio",herdingRatio,0.5);
	nhp.param<std::vector<float>>("bearWall",bearWall,{-4.0,4.0,-1.5,1.5});
	nhp.param<float>("height",height,1.0);
	nhp.param<float>("originRadius",originRadius,0.5);
	nhp.param<bool>("pushToGoal",pushToGoal,true);
	nhp.param<std::vector<float>>("xg",xg,{0.0,0.0});
	nhp.param<float>("velMaxRelDiff",velMaxRelDiff,0.1);
	nhp.param<float>("sm",sm,1.0);
	nhp.param<float>("s",s,0.1);
	nhp.param<int>("L1quad",L1quad,25);
	nhp.param<float>("lambda",lambda,0.001);
	nhp.param<bool>("useRobust",useRobust,false);

	int L1 = 4*L1quad;

	Bear bear(runNumber,bearName,sheepName,k1,k2,k3,ks,ky,kcl,Gamma,Deltat,N,herdingRatio,height,originRadius,bearWall,pushToGoal,xg,velMaxRelDiff,saveData,sm,s,L1,lambda,useRobust);

    ros::AsyncSpinner spinner(10);
    spinner.start();
    ros::waitForShutdown();
    //ros::spin();

    return 0;
}
