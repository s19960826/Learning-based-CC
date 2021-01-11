#ifndef MULTIPLE_FLOWS_H
#define MULTIPLE_FLOWS_H

#include "ns3/multiple-flows.h"
#include <vector>
#include <numeric>
#include <iostream>
#include <fstream>
#include <string>
#include <time.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/point-to-point-layout-module.h"
#include "ns3/applications-module.h"
#include "ns3/error-model.h"
#include "ns3/tcp-header.h"
#include "ns3/enum.h"
#include "ns3/event-id.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/ipv4-global-routing-helper.h"
#include "ns3/traffic-control-module.h"

#define TG_CDF_TABLE_ENTRY 32

namespace ns3
{

struct cdf_entry
{
    double value;
    double cdf;
};

/* CDF distribution */
struct cdf_table
{
    struct cdf_entry *entries;
    int num_entry;  /* number of entries in CDF table */
    int max_entry;  /* maximum number of entries in CDF table */
    double min_cdf; /* minimum value of CDF (default 0) */
    double max_cdf; /* maximum value of CDF (default 1) */
};

class MultipleFlows : public Object
{
    public:
	static TypeId GetTypeId (void);

	MultipleFlows();
	~MultipleFlows();

	ApplicationContainer InstallApplication(Ptr<Node> sendNode, Ptr<Node> sinkNode,
        Ipv4Address sourceAddress, Ipv4Address remoteAddress, uint16_t port, uint32_t maxBytes, uint32_t sendSize, uint32_t m_tuple, Ptr<ControlDecider> m_controller);
    	
	void InstallAllApplications(NodeContainer fromServers, NodeContainer destServers, double requestRate,
	std::vector<Ipv4Address> sourceAddresses, std::vector<Ipv4Address> destAddresses, 
	int flowCount, int start_port, int packet_size, Time timesim_start, Time timesim_end,Time time_flow_launch_end, Ptr<ControlDecider> m_controller);

	double poisson_gen_interval(int rate);

	double U_Random();

	uint32_t flow_size();

	uint32_t randTag = int(time(NULL));

		/* initialize a CDF distribution */
	static void init_cdf(struct cdf_table *table);

	/* free resources of a CDF distribution */
	static void free_cdf(struct cdf_table *table);

	/* get CDF distribution from a given file */
	static void load_cdf(struct cdf_table *table, const char *file_name);

	/* print CDF distribution information */
	static void print_cdf(struct cdf_table *table);

	/* get average value of CDF distribution */
	static double avg_cdf(struct cdf_table *table);

	/* Generate a random value based on CDF distribution */
	static double gen_random_cdf(struct cdf_table *table);

	static double interpolate(double x, double x1, double y1, double x2, double y2);

	/* generate a random floating point number from min to max */
	static double rand_range(double min, double max);

	static int rand_range(int min, int max);

	static double poission_gen_interval(double avg_rate);

};

} // namespace ns3


#endif
