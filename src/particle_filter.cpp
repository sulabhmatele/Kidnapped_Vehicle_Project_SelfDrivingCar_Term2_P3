/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <utility>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 42;

    default_random_engine gen;

    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i)
    {
        struct Particle part;
        part.x = dist_x(gen);
        part.y = dist_y(gen);
        part.theta = dist_theta(gen);
        part.weight = 1;

        particles.push_back(part);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for (int i = 0; i < num_particles; ++i)
    {
        double eff_yaw = yaw_rate * delta_t;
        double norm = 0;
        double eq_x = 0;
        double eq_y = 0;

        if (0 != yaw_rate)
        {
            norm = (velocity / yaw_rate);
            eq_x = norm * (sin(particles[i].theta + eff_yaw) - sin(particles[i].theta));
            eq_y = norm * (cos(particles[i].theta) - cos(particles[i].theta + eff_yaw));
        }
        else
        {
            eq_x = velocity * cos(particles[i].theta) * delta_t;
            eq_y = velocity * sin(particles[i].theta) * delta_t;
        }

        particles[i].x += eq_x;
        particles[i].y += eq_y;
        particles[i].theta += eff_yaw;

        normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
        normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
        normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);

        //angle normalization
        while (particles[i].theta> M_PI) particles[i].theta-=2.*M_PI;
        while (particles[i].theta<-M_PI) particles[i].theta+=2.*M_PI;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<UpdLandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    double shortest = 100; // Just random big number

    for (auto &observation : observations)
    {
        for (auto &landmark : predicted)
        {
            double tempdist = dist(observation.x, observation.y,
                                   landmark.x, landmark.y);

            if (tempdist < shortest)
            {
                shortest = tempdist;
                observation.id = landmark.id;
                observation.xl = landmark.x;
                observation.yl = landmark.y;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    std::vector<double> particles_wList; // Used for normalizing weights

    for(auto &single_part : particles)
    {
        /* Collect all the map landmarks in range from each particle */

        std::vector<LandmarkObs> predicted;

        for (auto &landmark : map_landmarks.landmark_list)
        {
            double tempdist = dist(single_part.x, single_part.y,
                                   landmark.x_f, landmark.y_f);

            if (tempdist < sensor_range)
            {
                struct LandmarkObs pred;
                pred.x = landmark.x_f;
                pred.y = landmark.y_f;
                pred.id = landmark.id_i;

                predicted.push_back(pred);
            }
        }

        /* Converting observations to map coordinate wrt this particle */

        std::vector<UpdLandmarkObs> trans_obs;
        double angle_displacement = 0; // Here we are assuming no angular shift

        for (auto &observation : observations)
        {
            struct UpdLandmarkObs lm;

            lm.x = single_part.x + cos(angle_displacement) * observation.x
                   - sin(angle_displacement) * observation.y;

            lm.y = single_part.y + sin(angle_displacement) * observation.x
                   + cos(angle_displacement) * observation.y;

            lm.id = observation.id;

            trans_obs.push_back(lm);
        }

        dataAssociation(predicted, trans_obs);

        /* Calculate weight of this particle */

        double weight = 0;
        double gauss_norm = (1/(2 * M_PI * std_landmark[0] * std_landmark[1]));

        for (auto single_obs:trans_obs)
        {
            double exponent = (pow((single_obs.x - single_obs.xl), 2) / (2 * pow(std_landmark[0],2)))
                               + (pow((single_obs.y - single_obs.yl), 2) / (2 * pow(std_landmark[1],2)));

            double obs_wt = gauss_norm * exp(-exponent);

            /* Final weight would be the multiplication of all the obs weights */
            weight *= obs_wt;
        }

        single_part.weight = weight;
        particles_wList.push_back(single_part.weight);
    }

    /* Normalizing weights to 0 - 1, we need to use them as probability in rsampling step */

    double min_weight = *std::min_element( std::begin(particles_wList), std::end(particles_wList) );
    double max_weight = *std::max_element( std::begin(particles_wList), std::end(particles_wList) );
    double diff_wt = (max_weight - min_weight);

    for(auto &single_part : particles)
    {
        single_part.weight = (single_part.weight - min_weight)/ diff_wt;
    }
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d({0,1});


}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= std::move(associations);
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
