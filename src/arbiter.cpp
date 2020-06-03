#include <iostream>
#include <cstdio>
#include <string>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ctime>

#include <ros/ros.h>
#include <sensor_msgs/Joy.h>
#include <sensor_msgs/JointState.h>
#include <std_msgs/Empty.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/AccelStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Int8.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

//differential q matrix
Eigen::MatrixXd getqDiff(Eigen::Vector4d q)
{
	Eigen::MatrixXd qDiff(4,3);
	qDiff << -q(1), -q(2), -q(3),
			  q(0), -q(3),  q(2),
			  q(3),  q(0), -q(1),
			 -q(2),  q(1),  q(0);
	return qDiff;
}

//q as matrix
Eigen::Matrix4d getqMat(Eigen::Vector4d q)
{
	Eigen::Matrix4d qMat;
	qMat << q(0), -q(1), -q(2), -q(3),
			q(1),  q(0), -q(3),  q(2),
			q(2),  q(3),  q(0), -q(1),
			q(3), -q(2),  q(1),  q(0);
	return qMat;
}

//q inverse
Eigen::Vector4d getqInv(Eigen::Vector4d q)
{
	Eigen::Vector4d qInv;
	qInv << q(0), -q(1), -q(2), -q(3);
	return qInv;
}

// trapizoidal rule integral estimator of a state vector wrt time
class IntegralEstimator
{
	std::deque<ros::Time> timeBuffer;        // time data
	std::deque<Eigen::VectorXd> stateBuffer; // state data
	int stateSize;                           // size of the state
	int bufferSize;

public:
	IntegralEstimator()
	{}

	IntegralEstimator(double bufferSizeInit, int stateSizeInit)
	{
		stateSize = stateSizeInit;
		bufferSize = bufferSizeInit;
	}

	IntegralEstimator(const IntegralEstimator& integralEstimatorNew)
	{
		stateSize = integralEstimatorNew.stateSize;
		bufferSize = integralEstimatorNew.bufferSize;
	}

	IntegralEstimator operator =(const IntegralEstimator& integralEstimatorNew)
	{
		stateSize = integralEstimatorNew.stateSize;
		bufferSize = integralEstimatorNew.bufferSize;
		return *this;
	}

	// update the buffers with new data
	Eigen::VectorXd update(Eigen::VectorXd stateNew, ros::Time timeNew)
	{
		timeBuffer.push_back(timeNew);   // save the time
		stateBuffer.push_back(stateNew); // save the state

		Eigen::VectorXd stateBufferIntegral;                    // integral of the buffer
		stateBufferIntegral = Eigen::VectorXd::Zero(stateSize); // initialize to 0

		// use greater than 3 because trapezoidal rule for 2 data points doesnt make sense
		if (timeBuffer.size() >= 3)
		{
			// while the buffer is too big pop off the oldest data as long as it wont make
			// the time on the buffer too small. compare with the second oldest data to ensure
			// the buffer stays large enough
			while (timeBuffer.size() > bufferSize)
			{
				timeBuffer.pop_front();
				stateBuffer.pop_front();
			}

			// if the buffer has enough time worth of data on it then calculate the
			// integral of Y, and calculate the new Dx
			if (timeBuffer.size() == bufferSize)
			{
				//for (int i = 0; i < timeBuffer.size()-1; i++)
				//{
					//stateBufferIntegral += 0.5*(timeBuffer.at(i+1) - timeBuffer.at(i)).toSec()*(stateBuffer.at(i+1) + stateBuffer.at(i));
				//}
				Eigen::VectorXd X = Eigen::VectorXd::Zero(stateSize*bufferSize);
				Eigen::MatrixXd H = Eigen::MatrixXd::Zero(stateSize*bufferSize,stateSize*2);
				Eigen::MatrixXd II = Eigen::MatrixXd::Identity(stateSize,stateSize);
				for (int i = 0; i < bufferSize; i++)
				{
					double timei = (timeBuffer.at(i) - timeBuffer.at(0)).toSec();
					int start = stateSize*i;
					X.segment(start,stateSize) = stateBuffer.at(i);
					Eigen::MatrixXd Hi(stateSize,2*stateSize);
					Hi << II, timei*II;
					H.block(start,0,stateSize,2*stateSize) = Hi;
				}
				//std::cout << "H\n" << H << std::endl;
				Eigen::MatrixXd HTH = H.transpose()*H;
				//std::cout << "HTH\n" << HTH << std::endl;
				Eigen::VectorXd HTX = H.transpose()*X;
				Eigen::VectorXd theta = HTH.ldlt().solve(HTX);
				//std::cout << "HTX\n" << HTX << std::endl;
				//std::cout << "theta\n" << theta << std::endl;
				double Dt = (timeBuffer.at(bufferSize-1) - timeBuffer.at(0)).toSec();
				stateBufferIntegral = theta.segment(0,stateSize) + Dt*theta.segment(stateSize,stateSize);
			}
		}
		//std::cout << "stateBufferIntegral\n" << stateBufferIntegral << std::endl;
		return stateBufferIntegral;
	}
};

// LS estimator for a first order approximatoion of the derivative of a state vector wrt time, thanks Anup
class DerivativeEstimator
{
    int bufferSize; //Number of data points to store
    int stateSize; //Number of elements for the state
    bool bufferFull; //Estimation will start after buffer is full for first time
    std::deque<ros::Time> timeBuff; //ring buffer for time data
    std::deque<Eigen::VectorXd> stateBuff; //ring buffer for position data
    bool firstUpdate;//indicates first update has not happened

public:
	DerivativeEstimator()
	{}

    DerivativeEstimator(int bufferSizeInit, int stateSizeInit)
    {
        //Initialize buffers
        bufferSize = bufferSizeInit;
        stateSize = stateSizeInit;
        bufferFull = false;
        firstUpdate = true;
    }

    DerivativeEstimator(const DerivativeEstimator& derivativeEstimatorNew)
    {
        //Initialize buffers
        bufferSize = derivativeEstimatorNew.bufferSize;
        stateSize = derivativeEstimatorNew.stateSize;
        bufferFull = false;
        firstUpdate = true;
    }

    DerivativeEstimator operator=(const DerivativeEstimator& derivativeEstimatorNew)
    {
        //Initialize buffers
        bufferSize = derivativeEstimatorNew.bufferSize;
        stateSize = derivativeEstimatorNew.stateSize;
        bufferFull = false;
        firstUpdate = true;
        return *this;
    }

    Eigen::VectorXd update(Eigen::VectorXd newMeasure, ros::Time newTime)
    {
		// Picture courtesy of Anup
        // Setting up least squares problem A*theta = P. theta is made up of the coefficients for the best fit line,
        // e.g., X = Mx*T + Bx, Y = My*t + By, Z = Mz*t + Bz. Velocity is estimated as the slope of the best fit line, i.e., Vx = Mx, Vy = My, Vz = Mz.
        // Each block of data is arranged like this:
        // [Xi]     [1, Ti,  0,  0,  0,  0] * [Bx]
        // [Yi]  =  [0,  0,  1, Ti,  0,  0]   [Mx]
        // [Zi]     [0,  0,  0,  0,  1, Ti]   [By]
        //  \/      \_____________________/   [My]
        //  Pi                 \/             [Bz]
        //                     Ai             [Mz]
        //                                     \/
        //                                   theta
        //
        // and then data is all stacked like this, where n is the buffer size:
        // [P1]     [A1] * [Bx]
        // [P2]  =  [A2]   [Mx]
        //  :        :     [By]
        // [Pn]     [An]   [My]
        //                 [Bz]
        //                 [Mz]

		firstUpdate = false;

        //Fill buffers
        timeBuff.push_back(newTime);
        stateBuff.push_back(newMeasure);

        //indicate if full
        if (!bufferFull && (timeBuff.size() >= bufferSize))
        {
			bufferFull = true;
		}

		//remove if too many
		while (timeBuff.size() > bufferSize)
		{
			timeBuff.pop_front();
			stateBuff.pop_front();
		}

        Eigen::VectorXd stateDerivative = Eigen::VectorXd::Zero(stateSize,1);//initialize state derivative
        if (bufferFull)
        {
            // normalize time for numerical stability/accuracy of subsequent matrix inversion

			clock_t startTime = clock();
            // Solve LLS for best fit line parameters
            Eigen::MatrixXd stateA = Eigen::MatrixXd::Zero(stateSize*bufferSize,2*stateSize);
            Eigen::VectorXd stateB = Eigen::VectorXd::Zero(stateSize*bufferSize);
            for (int ii = 0; ii < bufferSize; ii++)
            {
				Eigen::MatrixXd newA = Eigen::MatrixXd::Zero(stateSize,2*stateSize);
				float dtii = (timeBuff.at(ii) - timeBuff.at(0)).toSec();
				for (int jj = 0; jj < stateSize; jj++)
				{
					int thisColStart = 2*jj;
					newA.block(jj,thisColStart,1,2) << 1,dtii;
				}

				stateA.block(ii*stateSize,0,stateSize,2*stateSize) = newA;
				stateB.segment(ii*stateSize,stateSize) = stateBuff.at(ii);
            }
            //ROS_INFO("time for here6 %3.7f",double(clock()-startTime)/CLOCKS_PER_SEC);
            startTime = clock();
            Eigen::VectorXd theta = stateA.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(stateB);
			//Eigen::VectorXd theta = stateA.colPivHouseholderQr().solve(stateBuff);
			//Eigen::VectorXd theta = stateA.householderQr().solve(stateBuff);
			//Eigen::MatrixXd ATA = stateA.transpose()*stateA;
			//Eigen::MatrixXd ATB = stateA.transpose()*stateBuff;
			//Eigen::VectorXd theta = ATA.ldlt().solve(ATB);
			//Eigen::VectorXd theta = ATA.llt().solve(ATB);



			//ROS_INFO("time for here7 %3.7f",double(clock()-startTime)/CLOCKS_PER_SEC);

			// Get state derivatives
            for (int ii = 0; ii < stateSize; ii++)
            {
				int oddElement = ii*2+1;
				stateDerivative(ii) = theta(oddElement);
			}
        }

        return stateDerivative;//return state derivative
    }

	//return the current index in the state
	bool isfirstUpdate()
	{
		return firstUpdate;
	}

	//return if the buffer is full indicating the estimate is good
	bool isbufferFull()
	{
		return bufferFull;
	}

};

// PID controller for a state vector error
class PID
{
	double kP;
	double kD;
	double kI;
	int derivativeBufferSize;
	int integralBufferSize;
	DerivativeEstimator derivativeEstimator;
	IntegralEstimator integralEstimator;
	int stateSize;

public:
	PID()
	{}

	PID(double kPInit, double kDInit, double kIInit, int derivativeBufferSizeInit, int integralBufferSizeInit, int stateSizeInit)
	{
		kP = kPInit;
		kD = kDInit;
		kI = kIInit;
		stateSize = stateSizeInit;
		derivativeBufferSize = derivativeBufferSizeInit;
		integralBufferSize = integralBufferSizeInit;
		derivativeEstimator = DerivativeEstimator(derivativeBufferSize,stateSize);
		integralEstimator = IntegralEstimator(integralBufferSize,stateSize);
	}

	PID(const PID& pidNew)
	{
		kP = pidNew.kP;
		kD = pidNew.kD;
		kI = pidNew.kI;
		stateSize = pidNew.stateSize;
		derivativeBufferSize = pidNew.derivativeBufferSize;
		integralBufferSize = pidNew.integralBufferSize;
		derivativeEstimator = DerivativeEstimator(derivativeBufferSize,stateSize);
		integralEstimator = IntegralEstimator(integralBufferSize,stateSize);
	}

	PID operator =(const PID& pidNew)
	{
		kP = pidNew.kP;
		kD = pidNew.kD;
		kI = pidNew.kI;
		stateSize = pidNew.stateSize;
		derivativeBufferSize = pidNew.derivativeBufferSize;
		integralBufferSize = pidNew.integralBufferSize;
		derivativeEstimator = DerivativeEstimator(derivativeBufferSize,stateSize);
		integralEstimator = IntegralEstimator(integralBufferSize,stateSize);
		return *this;
	}

	Eigen::VectorXd update(Eigen::VectorXd errorNew, ros::Time timeNew)
	{
		Eigen::VectorXd kPu = kP*errorNew;
		Eigen::VectorXd kDu = kD*derivativeEstimator.update(errorNew, timeNew);
		Eigen::VectorXd kIu = kI*integralEstimator.update(errorNew, timeNew);
		return kPu+kDu+kIu;
	}

};

//surrounding ball for a robot
class Ball
{
	double R;//radius of ball
	double bodyMinRadius;//minimum radius for the ball
	double bodyGain;//gain for ball growth
	double bodyGrowthRate;//slope for ball growth
	double neighborGain;//gain for push from neighbor
	double neighborGrowthRate;//slope for neighbor
	double neighborTurnGain;//turn gain for neighbor
	Eigen::Vector3d pSW;//position of the body
	Eigen::Vector4d qSW;//orientation of the body
	Eigen::Vector3d vSW;//linear velocity of the body
	Eigen::Vector3d wSW;//angular velocity of the body
	Eigen::VectorXd walls;//walls
	double wallGain;//gain on wall
	double wallGrowthRate;//growth rate on wall
	double wallRadius;//offset of wall
	double wallTurnGain;//turn gain of wall

public:
	Ball()
	{}
	Ball(double bodyMinRadiusInit, double bodyGainInit, double bodyGrowthRateInit, double neighborGainInit, double neighborGrowthRateInit, double neighborTurnGainInit, std::vector<double> wallInit, double wallGainInit, double wallGrowthRateInit, double wallRadiusInit, double wallTurnGainInit)
	{
		R = bodyMinRadiusInit;
		bodyMinRadius = bodyMinRadiusInit;
		bodyGain = bodyGainInit;
		bodyGrowthRate = bodyGrowthRateInit;
		neighborGain = neighborGainInit;
		neighborGrowthRate = neighborGrowthRateInit;
		neighborTurnGain = neighborTurnGainInit;
		walls = Eigen::VectorXd::Zero(6);
		for (int i = 0; i < 6; i++) { walls(i) = wallInit.at(i); }
		wallGain = wallGainInit;
		wallGrowthRate = wallGrowthRateInit;
		wallRadius = wallRadiusInit;
		wallTurnGain = wallTurnGainInit;
	}

	Ball(const Ball& newBall)
	{
		R = newBall.R;
		bodyMinRadius = newBall.bodyMinRadius;
		bodyGain = newBall.bodyGain;
		bodyGrowthRate = newBall.bodyGrowthRate;
		neighborGain = newBall.neighborGain;
		neighborGrowthRate = newBall.neighborGrowthRate;
		neighborTurnGain = newBall.neighborTurnGain;
		walls = Eigen::VectorXd::Zero(6);
		for (int i = 0; i < 6; i++) { walls(i) = newBall.walls(i); }
		wallGain = newBall.wallGain;
		wallGrowthRate = newBall.wallGrowthRate;
		wallRadius = newBall.wallRadius;
		wallTurnGain = newBall.wallTurnGain;
	}

	Ball operator=(const Ball& newBall)
	{
		R = newBall.R;
		bodyMinRadius = newBall.bodyMinRadius;
		bodyGain = newBall.bodyGain;
		bodyGrowthRate = newBall.bodyGrowthRate;
		neighborGain = newBall.neighborGain;
		neighborGrowthRate = newBall.neighborGrowthRate;
		neighborTurnGain = newBall.neighborTurnGain;
		walls = Eigen::VectorXd::Zero(6);
		for (int i = 0; i < 6; i++) { walls(i) = newBall.walls(i); }
		wallGain = newBall.wallGain;
		wallGrowthRate = newBall.wallGrowthRate;
		wallRadius = newBall.wallRadius;
		wallTurnGain = newBall.wallTurnGain;
		return *this;
	}

	//update velocities
	//updates the ball radius given velocity of self wrt world in world
	void updateTwist(Eigen::Vector3d vSWNew, Eigen::Vector3d wSWNew)
	{
		vSW = vSWNew;
		wSW = wSWNew;
		R = bodyGain*std::tanh(bodyGrowthRate*vSW.norm())+bodyMinRadius;
	}

	//updates the position and orientation of self wrt world expressed in world
	void updatePose(Eigen::Vector3d pSWNew, Eigen::Vector4d qSWNew)
	{
		pSW = pSWNew;
		qSW = qSWNew;
	}

	//returns linear and angular velocity command from the wall
	Eigen::VectorXd wallVelCmd()
	{
		double xs = pSW(0);
		double ys = pSW(1);
		double zs = pSW(2);
		//std::cout << "pSW\n" << pSW << std::endl;
		double xmin = walls(0);
		double xmax = walls(1);
		double ymin = walls(2);
		double ymax = walls(3);
		double zmin = walls(4);
		double zmax = walls(5);
		//std::cout << "walls\n" << walls << std::endl;
		Eigen::VectorXd wallDiffs(6);
		wallDiffs << (xmin)-(xs-R),
					 (xs+R)-(xmax),
					 (ymin)-(ys-R),
					 (ys+R)-(ymax),
					 (zmin)-(zs-R),
					 (zs+R)-(zmax);
		//std::cout << "wallDiffs\n" << wallDiffs << std::endl;
		Eigen::VectorXd Ks = Eigen::VectorXd::Zero(6);
		for (int i = 0; i < 6; i++)
		{
			//std::cout << "wallGain\n" << wallGain << std::endl;
			//std::cout << "wallGrowthRate\n" << wallGrowthRate << std::endl;
			//std::cout << "wallDiffs(i)\n" << wallDiffs(i) << std::endl;
			//std::cout << "wallRadius\n" << wallRadius << std::endl;
			double K = wallGain*std::tanh(wallGrowthRate*wallDiffs(i)+wallRadius)+wallGain;
			//std::cout << "K\n" << K << std::endl;
			Ks(i) = K;
		}
		Eigen::Vector3d wallLinVel(Ks(0)-Ks(1),Ks(2)-Ks(3),Ks(4)-Ks(5));                                 				   						  					 // get the desired velocity vector from walls
		Eigen::Vector3d wallLinVelXY(wallLinVel(0),wallLinVel(1),0);                                     				   						 					 // get the xy desired velocity from the walls
		double wallLinVelXYAng = std::atan2(wallLinVelXY(1),wallLinVelXY(0));                            				   					      					 // get the xy direction of the desired velocity from walls
		Eigen::Vector4d wallLinVelQ(std::cos(wallLinVelXYAng/2.0), 0, 0, std::sin(wallLinVelXYAng/2.0)); 			  	   						  					 // wall orientation command
		if ((qSW - wallLinVelQ).norm() > (qSW - (-1.0*wallLinVelQ)).norm()) { wallLinVelQ *= -1.0; }      				   						  					 // flip the quaternion to the closest to the body
		Eigen::Vector4d qTilde = getqMat(wallLinVelQ)*qSW;                                                                 					      				     // get the error quaternion
		Eigen::Vector3d wallAngVelBody = -wallTurnGain*wallLinVelXY.norm()*Eigen::Vector3d(qTilde(1),qTilde(2),qTilde(3));                                           // get the angular command in the body
		Eigen::Vector3d wallAngVel = (getqMat(qSW)*getqMat(Eigen::Vector4d(0.0,wallAngVelBody(0),wallAngVelBody(1),wallAngVelBody(2)))*getqInv(qSW)).block(1,0,3,1); // get the angular command in the world
		Eigen::VectorXd wallVel(6);
		wallVel << wallLinVel,wallAngVel;
		return wallVel;
	}

	//returns linear velocity command from the neighbors
	Eigen::VectorXd neighborsVelCmd(Eigen::MatrixXd poses, Eigen::VectorXd radii, int numberNeighbors)
	{
		Eigen::VectorXd neighborVelSum = Eigen::VectorXd::Zero(6);
		Eigen::MatrixXd pSNs(3,numberNeighbors);
		Eigen::VectorXd Ks(numberNeighbors);
		for (int i = 0; i < numberNeighbors; i++)
		{
			Eigen::Vector3d pNW = poses.block(0,i,3,1);//neighbor position wrt world
			double RO = radii(i);//radius of neighbor
			Eigen::Vector3d pSN = pSW-pNW;
			double pSNNorm = pSN.norm();
			Eigen::Vector3d uSN = pSN/pSNNorm;
			pSNs.block(0,i,3,1) = pSN;
			double K = -1.0*std::tanh(neighborGrowthRate*pSNNorm - neighborGain*(RO+R))+1;
			//std::cout << "neighborGain\n" << neighborGain << std::endl;
			//std::cout << "pSNNorm\n" << pSNNorm << std::endl;
			//std::cout << "neighborGrowthRate\n" << neighborGrowthRate << std::endl;
			//std::cout << "RO\n" << RO << std::endl;
			//std::cout << "R\n" << R << std::endl;
			Ks(i) = K;

			Eigen::Vector3d neighborLinVel = K*uSN;
			Eigen::Vector3d neighborLinVelXY(neighborLinVel(0),neighborLinVel(1),0);																									 // get the xy desired velocity from the neighbor
			double neighborLinVelXYAng = std::atan2(neighborLinVelXY(1),neighborLinVelXY(0));																							 // get the xy direction of the desired velocity from neighbor
			Eigen::Vector4d neighborLinVelQ(std::cos(neighborLinVelXYAng/2.0), 0, 0, std::sin(neighborLinVelXYAng/2.0));																 // neighbor orientation command
			if ((qSW - neighborLinVelQ).norm() > (qSW - (-1.0*neighborLinVelQ)).norm()) { neighborLinVelQ *= -1.0; }																	 // flip the quaternion to the closest to the body
			Eigen::Vector4d qTilde = getqMat(neighborLinVelQ)*qSW;                                                                 					      				                 // get the error quaternion
			Eigen::Vector3d neighborAngVelBody = -neighborTurnGain*neighborLinVelXY.norm()*Eigen::Vector3d(qTilde(1),qTilde(2),qTilde(3));                                               // get the angular command in the body
			Eigen::Vector3d neighborAngVel = (getqMat(qSW)*getqMat(Eigen::Vector4d(0.0,neighborAngVelBody(0),neighborAngVelBody(1),neighborAngVelBody(2)))*getqInv(qSW)).block(1,0,3,1); // get the angular command in the world
			Eigen::VectorXd neighborVel(6);
			neighborVel << neighborLinVel,neighborAngVel;
			neighborVelSum += neighborVel;
		}
		return neighborVelSum;
	}

	double getradius()
	{
		return R;
	}
};

//handle for a bebop
class Bebop
{
	const std::string bebopName;//namespace for the bebop
	ros::NodeHandle nh;//nodehandle
	ros::Subscriber mocapPoseSub,desOdomSub;//returns the pose from mocap
	ros::Publisher gimbalAnglePub,takeoffPub,landPub,resetPub,cmdPub,odomPub,posePub,worldVelPub,bodyVelPub,bodyVelfrdPub;//gimbal angles,takeoff,land,reset,and command publishers, command publisher, odom publisher, posePub,body and world velocities, and body and world accelerations
	ros::Timer predictTimer;//watchdog to tell if lost line of sight and need to predict, upon recovery will need to reset
	geometry_msgs::Twist gimbalAngles;//pan and tilt angles
	double predictWatchdogTime;//time for the watchdog to wait before estimating the states from the approximates
	ros::Timer predictWatchdogTimer;//watchdog to tell if lost line of sight and need to predict
	bool inJoyMode;//indicates if the bebop has been selected for joystick control
	bool inAutoMode;//indicates if the bebop has been selected for autonomous control
	nav_msgs::Odometry odomWrtWorld;//holds odometry wrt world
	bool estimatorInitialized;//indicates the estimator has been initialized
	bool predicting;//indicates that predicting
	bool firstMocap;//indicates first mocap call
	int velocityBufferSize;//velocity buffer size
	DerivativeEstimator velocities;//LS velocities estimator
	ros::Time lastTime;//last message time
	Eigen::Vector3d lastPoseP;//last position
	Eigen::Vector4d lastPoseQ;//last orientation
	Eigen::Vector3d lastv;//last linear vel
	Eigen::Vector3d lastw;//last angular vel
	Eigen::Vector3d lastPoseMocapP;//last position from mocap
	Eigen::Vector4d lastPoseMocapQ;//last orientation from mocap
	bool newMeasure;//indicates a new measure has been received
	bool firstTimer;//indicates first timer call
	Ball ball;//surrounding ball
	Ball ballVel;//surrounding ball for self
	bool newDesired;//indicates a new desired has been received
	Eigen::Vector3d desPoseP;//holds desired position wrt world
	Eigen::Vector4d desPoseQ;//holds desired orientation wrt world
	Eigen::Vector3d lastdesPoseP;//holds desired position in world
	Eigen::Vector4d lastdesPoseQ;//holds desired orientation wrt world
	Eigen::Vector3d desLinVel;//holds desired linear velocity in world
	Eigen::Vector3d desAngVel;//holds desired angular velocity in world
	bool firstDesired;
	std::vector<double> posePGains,poseQGains,linVelGains,angVelGains;
	int errorDerivativeBufferSize;
	double errorIntegralBufferSize;
	PID posePPID;
	PID linVelPID;
	PID poseQPID;
	PID angVelPID;
	bool trackDesVel;
	bool trackWall;
	bool trackNeighbor;
	bool usePID;
	int bebopNum;

public:
	Bebop(std::string bebopTopic, double predictWatchdogTimeInit, int velocityBufferSizeInit, double bodyMinRadius, double bodyGain, double bodyGrowthRate, double neighborGain,
			  double neighborGrowthRate, double neighborTurnGain, std::vector<double> wallInit, double wallGain, double wallGrowthRate, double wallRadius,
			  double wallTurnGain, std::vector<double> posePGainsInit, std::vector<double> poseQGainsInit, std::vector<double> linVelGainsInit, std::vector<double> angVelGainsInit,
			  int errorDerivativeBufferSizeInit, int errorIntegralBufferSizeInit, bool trackDesVelInit, bool trackWallInit, bool trackNeighborInit, bool usePIDInit) : bebopName(bebopTopic)
	{
		//initialize everything for start
		nh = ros::NodeHandle(bebopName);
		mocapPoseSub = nh.subscribe("mocapPose",1,&Bebop::mocapPoseCB,this);
		gimbalAnglePub = nh.advertise<geometry_msgs::Twist>("camera_control",1);
		landPub = nh.advertise<std_msgs::Empty>("land",1);
		takeoffPub = nh.advertise<std_msgs::Empty>("takeoff",1);
		resetPub = nh.advertise<std_msgs::Empty>("reset",1);
		cmdPub = nh.advertise<geometry_msgs::Twist>("cmd_vel",1);
		odomPub = nh.advertise<nav_msgs::Odometry>("odometry",1);
		posePub = nh.advertise<geometry_msgs::PoseStamped>("mocapPoseFilt",1);
		worldVelPub = nh.advertise<geometry_msgs::TwistStamped>("mocapWorldVelFilt",1);
		bodyVelPub = nh.advertise<geometry_msgs::TwistStamped>("mocapBodyVelFilt",1);
		bodyVelfrdPub = nh.advertise<geometry_msgs::TwistStamped>("mocapBodyVelFRDFilt",1);
		desOdomSub = nh.subscribe("desOdom",1,&Bebop::desOdomCB,this);
		predictWatchdogTime = predictWatchdogTimeInit;
		predictWatchdogTimer = nh.createTimer(ros::Duration(predictWatchdogTime),&Bebop::predictWatchdogCB,this,false);
		//predictWatchdogTimer.stop();//stop it until the first mocap recieved
		inJoyMode = false;
		inAutoMode = false;
		odomWrtWorld.child_frame_id = bebopName;
		estimatorInitialized = false;
		predicting = false;
		firstMocap = true;
		velocityBufferSize = velocityBufferSizeInit;
		velocities = DerivativeEstimator(velocityBufferSize,7);
		newMeasure = false;
		firstTimer = true;
		ball = Ball(bodyMinRadius,bodyGain,bodyGrowthRate,neighborGain,neighborGrowthRate,neighborTurnGain,wallInit,wallGain,wallGrowthRate,wallRadius,wallTurnGain);
		newDesired = false;
		firstDesired = true;
		posePGains = posePGainsInit;
		poseQGains = poseQGainsInit;
		linVelGains = linVelGainsInit;
		angVelGains = angVelGainsInit;
		errorDerivativeBufferSize = errorDerivativeBufferSizeInit;
		errorIntegralBufferSize = errorIntegralBufferSizeInit;
		posePPID = PID(posePGains.at(0),posePGains.at(1),posePGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,3);
		poseQPID = PID(poseQGains.at(0),poseQGains.at(1),poseQGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,4);
		linVelPID = PID(linVelGains.at(0),linVelGains.at(1),linVelGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,3);
		angVelPID = PID(angVelGains.at(0),angVelGains.at(1),angVelGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,3);
		trackDesVel = trackDesVelInit;
		trackWall = trackWallInit;
		trackNeighbor = trackNeighborInit;
		usePID = usePIDInit;
	}

    Bebop(const Bebop& BebopN) : bebopName(BebopN.bebopName)
    {
		nh = ros::NodeHandle(bebopName);
		mocapPoseSub = nh.subscribe("mocapPose",1,&Bebop::mocapPoseCB,this);
		gimbalAnglePub = nh.advertise<geometry_msgs::Twist>("camera_control",1);
		landPub = nh.advertise<std_msgs::Empty>("land",1);
		takeoffPub = nh.advertise<std_msgs::Empty>("takeoff",1);
		resetPub = nh.advertise<std_msgs::Empty>("reset",1);
		cmdPub = nh.advertise<geometry_msgs::Twist>("cmd_vel",1);
		odomPub = nh.advertise<nav_msgs::Odometry>("odometry",1);
		posePub = nh.advertise<geometry_msgs::PoseStamped>("mocapPoseFilt",1);
		worldVelPub = nh.advertise<geometry_msgs::TwistStamped>("mocapWorldVelFilt",1);
		bodyVelPub = nh.advertise<geometry_msgs::TwistStamped>("mocapBodyVelFilt",1);
		bodyVelfrdPub = nh.advertise<geometry_msgs::TwistStamped>("mocapBodyVelFRDFilt",1);
		desOdomSub = nh.subscribe("desOdom",1,&Bebop::desOdomCB,this);
		predictWatchdogTime = BebopN.predictWatchdogTime;
		predictWatchdogTimer = nh.createTimer(ros::Duration(predictWatchdogTime),&Bebop::predictWatchdogCB,this,false);
		//predictWatchdogTimer.stop();//stop it until the first mocap recieved
		gimbalAngles = BebopN.gimbalAngles;
		inJoyMode = false;
		inAutoMode = false;
		odomWrtWorld.child_frame_id = bebopName;
		estimatorInitialized = false;
		predicting = false;
		firstMocap = true;
		velocityBufferSize = BebopN.velocityBufferSize;
		velocities = DerivativeEstimator(velocityBufferSize,7);
		newMeasure = false;
		firstTimer = true;
		ball = BebopN.ball;
		newDesired = false;
		firstDesired = true;
		posePGains = BebopN.posePGains;
		poseQGains = BebopN.poseQGains;
		linVelGains = BebopN.linVelGains;
		angVelGains = BebopN.angVelGains;
		errorDerivativeBufferSize = BebopN.errorDerivativeBufferSize;
		errorIntegralBufferSize = BebopN.errorIntegralBufferSize;
		posePPID = PID(posePGains.at(0),posePGains.at(1),posePGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,3);
		poseQPID = PID(poseQGains.at(0),poseQGains.at(1),poseQGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,4);
		linVelPID = PID(linVelGains.at(0),linVelGains.at(1),linVelGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,3);
		angVelPID = PID(angVelGains.at(0),angVelGains.at(1),angVelGains.at(2),errorDerivativeBufferSize,errorIntegralBufferSize,3);
		trackDesVel = BebopN.trackDesVel;
		trackWall = BebopN.trackWall;
		trackNeighbor = BebopN.trackNeighbor;
		usePID = BebopN.usePID;
    }

	void land()//command to tell this bebop to land
	{
		landPub.publish(std_msgs::Empty());
	}

	void takeoff()//command to tell this bebop to takeoff
	{
		takeoffPub.publish(std_msgs::Empty());
	}

	void kill()//command to tell this bebop to kill its motors
	{
		resetPub.publish(std_msgs::Empty());
	}

	void updateGimbal(double panAngle, double tiltAngle)// update the pan and tilt angles of this bebops gimbal
	{
		gimbalAngles.angular.y += tiltAngle;
		if (gimbalAngles.angular.y > 17.0) { gimbalAngles.angular.y = 17.0; }
		if (gimbalAngles.angular.y < -35.0) { gimbalAngles.angular.y = -35.0; }
		gimbalAngles.angular.z += panAngle;
		gimbalAnglePub.publish(gimbalAngles);
	}

	//get desired odom
	void desOdomCB(const nav_msgs::Odometry::ConstPtr& msg)
	{
		desPoseP = Eigen::Vector3d(msg->pose.pose.position.x,msg->pose.pose.position.y,msg->pose.pose.position.z);
		desPoseQ = Eigen::Vector4d(msg->pose.pose.orientation.w,msg->pose.pose.orientation.x,msg->pose.pose.orientation.y,msg->pose.pose.orientation.z);
		desLinVel = Eigen::Vector3d(msg->twist.twist.linear.x,msg->twist.twist.linear.y,msg->twist.twist.linear.z);
		desAngVel = Eigen::Vector3d(msg->twist.twist.angular.x,msg->twist.twist.angular.y,msg->twist.twist.angular.z);

		if (firstDesired)
		{
			lastdesPoseQ = desPoseQ;
			firstDesired = false;
		}
		else
		{
			if ( (lastdesPoseQ - desPoseQ).norm() > (lastdesPoseQ - (-1.0*desPoseQ)).norm() ) { desPoseQ *= -1.0; }
		}

		newDesired = true;
		//std::cout << "hey i got a odom\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";
	}

	//callback for the mocap
	void mocapPoseCB(const geometry_msgs::PoseStamped::ConstPtr& msg)
	{
		//predictWatchdogTimer.stop();//stop the timer
		Eigen::Vector3d newPoseP(3,1);//use fresh store
		Eigen::Vector4d newPoseQ(4,1);//use fresh store
		newPoseP << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
		newPoseQ << msg->pose.orientation.w, msg->pose.orientation.x, msg->pose.orientation.y, msg->pose.orientation.z;
		if (firstMocap)//indicate the first measure has been received
		{
			firstMocap = false;
		}
		lastPoseMocapP = newPoseP;//save measure
		lastPoseMocapQ = newPoseQ;//save measure
		newMeasure = true;//indicate a new measure has been received
		//predictWatchdogTimer.start();//start the timer
	}

	//clears joy mode and auto mode
	void clearMode()
	{
		inJoyMode = false;
		inAutoMode = false;
		newDesired = false;
	}

	//updates joy mode state
	void setinJoyMode(bool newJoyModeState)
	{
		if (newJoyModeState) { inAutoMode = false; }
		inJoyMode = newJoyModeState;
		newDesired = false;
	}

	//updates autonomous mode state
	void setinAutoMode(bool newAutoModeState)
	{
		inAutoMode = newAutoModeState;
		newDesired = false;
	}

	//returns indicator of joy mode state
	bool getinJoyMode()
	{
		return inJoyMode;
	}

	//returns this autonomous mode state
	bool getinAutoMode()
	{
		return inAutoMode;
	}

	// update the velocity command
	void updateVel(double vxJoy, double vyJoy, double vzJoy, double wzJoy, Eigen::MatrixXd poses, Eigen::VectorXd radii, int numberNeighbors, double kw, bool aButton, bool bButton, bool xButton, bool yButton, double panAngle, double tiltAngle)
	{
		//std::cout << std::endl;
		//std::cout << std::endl;
		//std::cout << std::endl;
		//std::cout << bebopName << std::endl;
		//std::cout << std::endl;

		double dt = 0.01; // time step for the wall and neighbor commands to move position

		Eigen::Vector3d desPosePCmd;
		Eigen::Vector4d desPoseQCmd;
		Eigen::Vector3d desLinVelCmd;
		Eigen::Vector3d desAngVelCmd;

		// wall commands
		Eigen::VectorXd wallVelCmd(6);
		wallVelCmd = ball.wallVelCmd();
		Eigen::Vector3d wallLinVelCmd = wallVelCmd.segment(0,3);
		Eigen::Vector3d wallAngVelCmd = wallVelCmd.segment(3,3);

		// neighbor commands
		Eigen::VectorXd neighborsVelCmd(6);
		neighborsVelCmd = ball.neighborsVelCmd(poses,radii,numberNeighbors);//get the neighbor command in the world
		Eigen::Vector3d neighborLinVelCmd = neighborsVelCmd.segment(0,3);
		Eigen::Vector3d neighborAngVelCmd = neighborsVelCmd.segment(3,3);

		//commands from the xbox controller
		if (inJoyMode)
		{
			std::cout << bebopName << " is in joymode\n";
			desPosePCmd = lastPoseP;
			desPoseQCmd = lastPoseQ;
			desLinVelCmd = (getqMat(lastPoseQ)*getqMat(Eigen::Vector4d(0.0,vxJoy,vyJoy,vzJoy))*getqInv(lastPoseQ)).block(1,0,3,1);
			desAngVelCmd = (getqMat(lastPoseQ)*getqMat(Eigen::Vector4d(0.0,0.0,0.0,wzJoy))*getqInv(lastPoseQ)).block(1,0,3,1);
			if (aButton)//if a button land
			{
				land();
				std::cout << bebopName << " is in land\n";
				return;
			}
			if (xButton)//if x button set or unset inAutoMode
			{
				//bool newAutoMode = false;
				//if (!inAutoMode) { newAutoMode = true; }
				std::cout << bebopName << " is in automode\n";
				newDesired = false;
				setinAutoMode(true);
			}
			if (yButton)//if y button takeoff
			{
				takeoff();
				std::cout << bebopName << " is in takeoff\n";
				return;
			}
			if (bButton)//if b button kill
			{
				kill();
			}
		}

		// commands from the paper controller
		if (inAutoMode)
		{
			if (newDesired)
			{
				desPosePCmd = desPoseP;
				desPoseQCmd = desPoseQ;
				desLinVelCmd = desLinVel;
				desAngVelCmd = desAngVel;
			}
			else
			{
				desPosePCmd = lastPoseP;
				desPoseQCmd = lastPoseQ;
				desLinVelCmd = Eigen::Vector3d(0,0,0);
				desAngVelCmd = Eigen::Vector3d(0,0,0);
			}
		}

		// commands to station keep
		if (!inAutoMode && !inJoyMode)
		{
			desPosePCmd = lastPoseP;
			desPoseQCmd = lastPoseQ;
			desLinVelCmd = Eigen::Vector3d(0,0,0);
			desAngVelCmd = Eigen::Vector3d(0,0,0);
		}

		ros::Time timeNew = ros::Time::now();
		Eigen::Vector3d posePOut = desPosePCmd;
		Eigen::Vector4d poseQOut = desPoseQCmd;
		Eigen::Vector3d linVelOut = Eigen::Vector3d::Zero();
		Eigen::Vector3d angVelOut = Eigen::Vector3d::Zero();

		if (trackWall)
		{
			linVelOut += wallLinVelCmd;
			angVelOut += wallAngVelCmd;
		}

		if (trackNeighbor)
		{
			linVelOut += neighborLinVelCmd;
			angVelOut += neighborAngVelCmd;
		}

		if (trackDesVel)
		{
			linVelOut += desLinVelCmd;
			angVelOut += desAngVelCmd;
		}
		Eigen::Vector3d posePError = posePOut - lastPoseP;

		Eigen::Vector4d poseQOutN = -1.0*poseQOut;
		if ((lastPoseQ - poseQOutN).norm() < (lastPoseQ - poseQOut).norm())
		{
			poseQOut = poseQOutN;
		}

		Eigen::Vector4d poseQError = getqMat(getqInv(poseQOut))*lastPoseQ;

		Eigen::Vector3d linVelError = linVelOut - lastv;
		Eigen::Vector3d angVelError = angVelOut - lastw;

		Eigen::Vector3d posePCmd = posePPID.update(posePError,timeNew);
		Eigen::Vector4d poseQCmd = poseQPID.update(poseQError,timeNew);
		Eigen::Vector3d linVelCmd = linVelPID.update(linVelError,timeNew);
		Eigen::Vector3d angVelCmd = angVelPID.update(angVelError,timeNew);

		//Eigen::Vector3d linCmdOut = posePCmd + linVelCmd + linVelOut/9.0;
		Eigen::Vector3d linCmdOut = posePCmd + linVelCmd + linVelOut;
		Eigen::Vector3d angCmdOut = -1.0*poseQCmd.segment(1,3) + angVelCmd + angVelOut;
		//Eigen::Vector3d linCmdOut = posePCmd + linVelCmd;
		//Eigen::Vector3d angCmdOut = -1.0*poseQCmd.segment(1,3) + angVelCmd;

		Eigen::Vector3d linCmdOutBody = (getqMat(getqInv(lastPoseQ))*getqMat(Eigen::Vector4d(0.0,linCmdOut(0),linCmdOut(1),linCmdOut(2)))*lastPoseQ).block(1,0,3,1);
		Eigen::Vector3d angCmdOutBody = (getqMat(getqInv(lastPoseQ))*getqMat(Eigen::Vector4d(0.0,angCmdOut(0),angCmdOut(1),angCmdOut(2)))*lastPoseQ).block(1,0,3,1);

		//if (bebopName == "bebop3")
		//{
			//std::cout << bebopName << std::endl;
			//std::cout << "desPosition" << std::endl;
			//std::cout << posePOut << std::endl;
		//}
		geometry_msgs::Twist cmdMsg;

		if (usePID)
		{
			cmdMsg.linear.x = linCmdOutBody(0);
			cmdMsg.linear.y = linCmdOutBody(1);
			cmdMsg.linear.z = linCmdOutBody(2);
			cmdMsg.angular.z = angCmdOutBody(2);
		}
		else
		{
			cmdMsg.linear.x = vxJoy;
			cmdMsg.linear.y = vyJoy;
			cmdMsg.linear.z = vzJoy;
			cmdMsg.angular.z = wzJoy;
		}
		cmdPub.publish(cmdMsg);
	}

	//timer update to increase and maintain continuity in state estimate
	void predictWatchdogCB(const ros::TimerEvent& event)
	{
		ros::Time startTime = ros::Time::now();
		//clock_t startTime = clock();
		//predictWatchdogTimer.stop();//stop the timer
		if (firstMocap) { return; }//if the first measure has not been received return
		if (firstTimer && !newMeasure) { return; }//shouldnt happen but just in case
		if (firstTimer)//if this is the first time the timer made it past the mocap block then initialize orientation and return
		{
			lastTime = event.current_real;
			lastPoseP = lastPoseMocapP;
			lastPoseQ = lastPoseMocapQ;

			if (lastPoseQ(0) < 0.0)
			{
				lastPoseQ *= -1.0;
			}

			firstTimer = false;
			newMeasure = false;
			//predictWatchdogTimer.start();//start the timer
			return;
		}

		lastTime = event.current_real;//update last time

		Eigen::Vector3d newPoseP;
		Eigen::Vector4d newPoseQ;
		Eigen::VectorXd newPose(7);
		if (newMeasure)//if new measure save it for use
		{
			newPoseP = lastPoseMocapP;
			newPoseQ = lastPoseMocapQ;
			if ((lastPoseQ-(-1*newPoseQ)).norm() < (lastPoseQ-newPoseQ).norm())//check if it flipped
			{
				newPoseQ *= -1;
			}
			newMeasure = false;
		}
		else
		{
			newPoseP = lastPoseP;
			newPoseQ = lastPoseQ;
		}
		newPose << newPoseP,newPoseQ;//save the measure

		Eigen::VectorXd lastVel = velocities.update(newPose,ros::Time::now());//update velocity
		Eigen::Vector3d lastLinVel = lastVel.segment(0,3);
		Eigen::Vector4d lastqDot = lastVel.segment(3,4);
		Eigen::MatrixXd Bq = getqDiff(lastPoseQ);//get the differential matrix for new orientation
		Eigen::Vector3d lastAngVelBody = 2*(Bq.transpose())*lastqDot;//get the angular velocity in the body frame
		lastPoseP = newPoseP;
		lastPoseQ = newPoseQ;
		lastv = lastLinVel;
		lastw = (getqMat(getqMat(lastPoseQ)*Eigen::Vector4d(0.0,lastAngVelBody(0),lastAngVelBody(1),lastAngVelBody(2)))*getqInv(lastPoseQ)).block(1,0,3,1);

		if (velocities.isbufferFull())//if the velocity buffer is running indicate it started
		{
			if (!estimatorInitialized)
			{
				estimatorInitialized = true;
			}
		}

		//get data in body frame
		Eigen::Vector3d lastvBody = (getqMat(getqMat(getqInv(lastPoseQ))*Eigen::Vector4d(0.0,lastv(0),lastv(1),lastv(2)))*lastPoseQ).block(1,0,3,1);
		Eigen::Vector3d lastwBody = lastAngVelBody;

		//messages
		geometry_msgs::PoseStamped poseMsg;
		geometry_msgs::TwistStamped velMsg;
		geometry_msgs::TwistStamped velBodyMsg;
		geometry_msgs::TwistStamped velBodyfrdMsg;

		//set stamp and frame id
		poseMsg.header.stamp = lastTime;
		poseMsg.header.frame_id = "world";
		velMsg.header.stamp = lastTime;
		velBodyMsg.header.stamp = lastTime;
		velBodyfrdMsg.header.stamp = lastTime;
		odomWrtWorld.header.stamp = lastTime;
		odomWrtWorld.header.frame_id = "world";

		//set data
		poseMsg.pose.position.x = lastPoseP(0); poseMsg.pose.position.y = lastPoseP(1); poseMsg.pose.position.z = lastPoseP(2);
		poseMsg.pose.orientation.w = lastPoseQ(0); poseMsg.pose.orientation.x = lastPoseQ(1); poseMsg.pose.orientation.y = lastPoseQ(2); poseMsg.pose.orientation.z = lastPoseQ(3);

		velMsg.twist.linear.x = lastv(0); velMsg.twist.linear.y = lastv(1); velMsg.twist.linear.z = lastv(2);
		velMsg.twist.angular.x = lastw(0); velMsg.twist.angular.y = lastw(1); velMsg.twist.angular.z = lastw(2);

		velBodyMsg.twist.linear.x = lastvBody(0); velBodyMsg.twist.linear.y = lastvBody(1); velBodyMsg.twist.linear.z = lastvBody(2);
		velBodyMsg.twist.angular.x = lastwBody(0); velBodyMsg.twist.angular.y = lastwBody(1); velBodyMsg.twist.angular.z = lastwBody(2);

		velBodyfrdMsg.twist.linear.x = lastvBody(0); velBodyfrdMsg.twist.linear.y = -lastvBody(1); velBodyfrdMsg.twist.linear.z = -lastvBody(2);
		velBodyfrdMsg.twist.angular.x = lastwBody(0); velBodyfrdMsg.twist.angular.y = -lastwBody(1); velBodyfrdMsg.twist.angular.z = -lastwBody(2);

		odomWrtWorld.pose.pose = poseMsg.pose;
		odomWrtWorld.twist.twist = velBodyMsg.twist;

		//publish
		posePub.publish(poseMsg);
		worldVelPub.publish(velMsg);
		bodyVelPub.publish(velBodyMsg);
		bodyVelfrdPub.publish(velBodyfrdMsg);
		odomPub.publish(odomWrtWorld);

		//update ball
		ball.updateTwist(lastv, lastw);//update velocities for ball
		ball.updatePose(lastPoseP, lastPoseQ);//update pose for ball

		//predictWatchdogTimer.start();//start timer
		//ROS_INFO("timer time %3.7f",double(clock()-startTime)/CLOCKS_PER_SEC);
		//ROS_INFO("timer time %3.7f",double((ros::Time::now()-startTime).toSec()));
	}
	Eigen::Vector3d getlastPoseP()
	{
		return lastPoseP;
	}
	Eigen::Vector4d getlastPoseQ()
	{
		return lastPoseQ;
	}
	Eigen::Vector3d getlastv()
	{
		return lastv;
	}
	Eigen::Vector3d getlastw()
	{
		return lastw;
	}
	bool getestimatorInitialized()
	{
		return estimatorInitialized;
	}
	double getradius()
	{
		return ball.getradius();
	}
};

//main class
class Arbiter
{
	ros::NodeHandle nh;
	ros::Subscriber joySub;
	bool newJoy;//indicates a new joy has been received
	std::vector<double> joyCmdBebopGain;//joystick command bebop gain
	std::vector<Bebop> bebops;//active bebops
	double kw;//angular velocity command gain on orientation error

public:
	Arbiter()
	{
		//handle to launch file parameters
		ros::NodeHandle nhp("~");

		bool trackDesVel,trackWall,trackNeighbor,usePID;
		nhp.param<bool>("trackDesVel",trackDesVel,true);
		nhp.param<bool>("trackWall",trackWall,true);
		nhp.param<bool>("trackNeighbor",trackNeighbor,true);
		nhp.param<bool>("usePID",usePID,true);

		//bebop specific params
		std::vector<std::string> activeBebops;
		nhp.param<std::vector<double>>("joyCmdBebopGain",joyCmdBebopGain,{0.1,0.1,0.1,0.1});
		nhp.param<std::vector<std::string>>("activeBebops",activeBebops,{"none"});
		double bodyMinRadiusBebop, bodyMaxRadiusBebop, bodyGainBebop, bodyGrowthRateBebop;
		double neighborGainBebop, neighborGrowthRateBebop, neighborTurnGainBebop;
		nhp.param<double>("bodyMinRadiusBebop",bodyMinRadiusBebop,0.75);
		nhp.param<double>("bodyMaxRadiusBebop",bodyMaxRadiusBebop,1.0);
		bodyGainBebop = bodyMaxRadiusBebop-bodyMinRadiusBebop;
		nhp.param<double>("bodyGrowthRateBebop",bodyGrowthRateBebop,2.0);
		nhp.param<double>("neighborGainBebop",neighborGainBebop,1.0);
		nhp.param<double>("neighborGrowthRateBebop",neighborGrowthRateBebop,2.0);
		nhp.param<double>("neighborTurnGainBebop",neighborTurnGainBebop,0.3);
		double wallGainBebop, wallGrowthRateBebop, wallRadiusBebop, wallTurnGainBebop;
		nhp.param<double>("wallGainBebop",wallGainBebop,1.0);
		nhp.param<double>("wallGrowthRateBebop",wallGrowthRateBebop,12.0);
		nhp.param<double>("wallRadiusBebop",wallRadiusBebop,-10.0);
		nhp.param<double>("wallTurnGainBebop",wallTurnGainBebop,2.0);

		// PID params
		std::vector<double> posePGains,poseQGains,linVelGains,angVelGains;
		nhp.param<std::vector<double>>("posePGains",posePGains,{0,0,0});
		nhp.param<std::vector<double>>("poseQGains",poseQGains,{0,0,0});
		nhp.param<std::vector<double>>("linVelGains",linVelGains,{0,0,0});
		nhp.param<std::vector<double>>("angVelGains",angVelGains,{0,0,0});

		int errorDerivativeBufferSize, errorIntegralBufferSize;
		nhp.param<int>("errorDerivativeBufferSize",errorDerivativeBufferSize,20);
		nhp.param<int>("errorIntegralBufferSize",errorIntegralBufferSize,20);

		//filter params
		nhp.param<double>("kw",kw,0.1);
		int velocityBufferSizeInit;
		nhp.param<int>("velocityBufferSizeInit",velocityBufferSizeInit,20);
		double predictWatchdogTimeInit;
		nhp.param<double>("predictWatchdogTimeInit",predictWatchdogTimeInit,0.001);
		std::vector<double> wallInit;
		nhp.param<std::vector<double>>("wallInit",wallInit,{-3.0,3.0,-1.5,1.5,0.5,3.0});

		//intialize the bebops
		bool noBebops = true;
		if (activeBebops.at(0).compare("none") != 0) { noBebops = false; }//check if no bebops activated
		if (!noBebops)//if there are active bebops will initialize them
		{
			for (int i = 0; i < activeBebops.size(); i++)
			{
				bebops.push_back(Bebop(activeBebops.at(i),predictWatchdogTimeInit,velocityBufferSizeInit,bodyMinRadiusBebop,bodyGainBebop,bodyGrowthRateBebop,
				                       neighborGainBebop,neighborGrowthRateBebop,neighborTurnGainBebop,wallInit,wallGainBebop,wallGrowthRateBebop,wallRadiusBebop,
				                       wallTurnGainBebop,posePGains,poseQGains,linVelGains,angVelGains,errorDerivativeBufferSize,errorIntegralBufferSize,trackDesVel,
				                       trackWall,trackNeighbor,usePID));
			}
		}

		if (noBebops) { ROS_ERROR("no platforms selected\n"); ros::shutdown(); }

		joySub = nh.subscribe("joy",1,&Arbiter::joyCB,this);//subscribe to the joystick

	}

	//callback to receive new joy message
	void joyCB(const sensor_msgs::Joy::ConstPtr& msg)
	{
		//std::cout << "hi" << std::endl;
		bool aButton = msg->buttons[0]>0;//a button
		bool bButton = msg->buttons[1]>0;//b button
		bool xButton = msg->buttons[2]>0;//x button
		bool yButton = msg->buttons[3]>0;//y button
		bool leftBumper = msg->buttons[4]>0;//left bumper
		bool rightBumper = msg->buttons[5]>0;//right bumper
		bool backButton = msg->buttons[6]>0;//back button
		bool startButton = msg->buttons[7]>0;//start button
		bool centerButton = msg->buttons[8]>0;//center button
		bool dpadL = msg->buttons[11]>0;//dpad left value
		bool dpadR = msg->buttons[12]>0;//dpad right value
		bool dpadU = msg->buttons[13]>0;//dpad up value
		bool dpadD = msg->buttons[14]>0;//dpad down value
		double leftStickHorz = msg->axes[0];//left thumbstick horizontal
		double leftStickVert = msg->axes[1];//left thumbstick vertical
		double leftTrigger = msg->axes[2];//left trigger
		double rightStickHorz = msg->axes[3];//right thumbstick horizontal
		double rightStickVert = msg->axes[4];//right thumbstick vertical
		double rightTrigger = msg->axes[5];//right trigger

		int joyModifyGroup = 0;//indicates which group to modify, neither, specific bebop, or all bebops
		if (!rightBumper && !leftBumper) { joyModifyGroup = 0; }//if neither bumper pulled no modification desired
		if (rightBumper && !leftBumper) { joyModifyGroup = 1; }//if right bumper and not left modify bebops
		if (rightBumper && leftBumper) { joyModifyGroup = 2; }//if left bumper and right bumper modify everything

		int joyModifyNumber = -1;//indicates which number to modify in the group
		if (startButton) { joyModifyNumber = 0; }//modify leader
		if (dpadU) { joyModifyNumber = 1; }//modify 1
		if (dpadR) { joyModifyNumber = 2; }//modify 2
		if (dpadD) { joyModifyNumber = 3; }//modify 3
		if (dpadL) { joyModifyNumber = 4; }//modify 4

		//if center button is pressed will clear all and return
		if (centerButton)
		{
			for (int i = 0; i < bebops.size(); i++) { bebops.at(i).clearMode(); }
		}

		//modify a member of a group if it is selected
		switch (joyModifyGroup)
		{
			case 0://modify nothing
			{
				break;
			}
			case 1://modify specific bebops
			{
				if (joyModifyNumber >= 0)//if less than or equal to 0 modify nothing
				{
					for (int i = 0; i < bebops.size(); i++)//go through the bebops and if the joy modify number is the index set it to joy mode
					{
						bool inJoyMode = false;
						if (i==joyModifyNumber)
						{
							inJoyMode = true;
						}
						bebops.at(i).setinJoyMode(inJoyMode);
					}
				}
				break;
			}
			case 2://modify all bebops
			{
				if (backButton)//if back button clear all from joy mode
				{
					for (int i = 0; i < bebops.size(); i++)//go through the bebops and set them to not be in joy mode
					{
						bebops.at(i).setinJoyMode(false);
					}
				}
				if (startButton)//if start button put all in joy mode
				{
					for (int i = 0; i < bebops.size(); i++)//go through the bebops and set them to be in joy mode
					{
						bebops.at(i).setinJoyMode(true);
					}
				}
				break;
			}
		}

		//pan and tilt adjustment
		double panAngle = 0;
		double tiltAngle = 0;
		if (dpadU && !(rightBumper || leftBumper) ) { tiltAngle = 0.5; }//if no bumpers are pulled and the up dpad is pressed, update gimbal angle up
		if (dpadD && !(rightBumper || leftBumper) ) { tiltAngle = -0.5; }//if no bumpers are pulled and the down dpad is pressed, update gimbal angle down

		//manual velocity adjustment
		double vxBebop = joyCmdBebopGain.at(0)*leftStickVert;
		double vyBebop = joyCmdBebopGain.at(1)*leftStickHorz;
		double vzBebop = joyCmdBebopGain.at(2)*rightStickVert;
		double wzBebop = joyCmdBebopGain.at(3)*rightStickHorz;

		//get all the current lastest poses and velocitites in the world and body frames for any bebop that has had its estimator initialized
		int numberBebops = 0;
		for (int i = 0; i < bebops.size(); i++) { if (bebops.at(i).getestimatorInitialized()) { numberBebops++; } }
		if (numberBebops < bebops.size()) { return; }//if all the bebops have not had their estimators started return

		//get bebops positions and radii
		Eigen::MatrixXd bebopPosesP(3,numberBebops);
		Eigen::VectorXd bebopradii(numberBebops);
		for (int i = 0; i < numberBebops; i++)
		{
			Eigen::Vector3d bebopPoseP = bebops.at(i).getlastPoseP();
			bebopPosesP.block(0,i,3,1) = bebopPoseP;
			bebopradii(i) = bebops.at(i).getradius();
		}

		//send in neighbor information, controller information, and teleop information to joy command to let bebops either use joy command, use desired commands, or hold pose and avoid neighbors
		for (int i = 0; i < bebops.size(); i++)
		{
			Eigen::MatrixXd PosesPThis(3,numberBebops-1);
			Eigen::VectorXd radiiThis(numberBebops-1);

			int index = 0;
			for (int j = 0; j < numberBebops; j++)
			{
				if (j != i)
				{
					PosesPThis.block(0,index,3,1) = bebopPosesP.block(0,j,3,1);
					radiiThis(index) = bebopradii(j);
					index++;
				}
			}
			bebops.at(i).updateVel(vxBebop,vyBebop,vzBebop,wzBebop,PosesPThis,radiiThis,numberBebops-1,kw,aButton,bButton,xButton,yButton,panAngle,tiltAngle);//update the velocity command

			if (bebops.at(i).getinJoyMode())
			{
				bebops.at(i).updateGimbal(panAngle, tiltAngle);
			}
		}
	}

};

int main(int argc, char** argv)
{
	ros::init(argc,argv,"arbiter");

	Arbiter arbiter;

	ros::AsyncSpinner spinner(4); // Use 4 threads
    spinner.start();
    ros::waitForShutdown();
	ros::spin();

    return 0;
}
