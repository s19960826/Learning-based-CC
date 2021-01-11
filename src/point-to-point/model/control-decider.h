#ifndef CONTROL_DECIDER_H
#define CONTROL_DECIDER_H

#include "ns3/application.h"
#include <vector>
#include <numeric>
#include <map>

namespace ns3

{
class ControlDecider : public Object
{
public:
	static TypeId GetTypeId (void);

	ControlDecider();
	~ControlDecider();

	//flow classification
	void mapTuple(uint32_t m_tuple, uint64_t packetSize);
	void finalTuple(uint32_t m_tuple);
  	void classifyFlows(uint32_t m_tuple);
	std::map<uint32_t, std::vector<uint64_t>> tmp_mapTuple;
	std::map<uint32_t, std::vector<uint64_t>> final_mapTuple;
	std::map<uint32_t, std::vector<uint64_t>> m_type;
	int randtmp = 1;

    
	//throughput and RTT
	void SendTime(uint32_t m_tuple,uint64_t sendValue);
	std::map<uint32_t, std::vector<uint64_t>> m_sendTime;

	void RecvTime(uint32_t m_tuple,uint64_t recvValue);
	std::map<uint32_t, std::vector<uint64_t>> m_recvTime;

	void IdTime(uint32_t m_tuple,uint64_t idtime);
	void ReTime(uint32_t m_tuple,uint64_t retime);
	void IdValue(uint32_t m_tuple,uint64_t idvalue);

	std::map<uint32_t, std::vector<uint64_t>>  idTime;
	std::map<uint32_t, std::vector<uint64_t>> reTime;
	std::map<uint32_t, std::vector<uint64_t>>  idValue;

	double AvgThroughput(uint32_t m_tuple,uint64_t rtt, uint64_t time);
	double avgThroughput;
	int loc;

	double LostRate(uint32_t m_tuple,uint64_t rtt, uint64_t time);
	double lostRate;

	double AvgRTT(uint32_t m_tuple, std::vector<uint64_t>, std::vector<uint64_t>, uint64_t rtt, uint64_t time);
	double avgRtt;



	/*
	void RecvTime(uint32_t m_tuple,uint64_t recvValue);

    void AvgTime(uint32_t m_tuple,uint64_t rtt);	
	std::map<uint32_t, std::vector<uint64_t>> m_sendTime;
	std::map<uint32_t, std::vector<uint64_t>> m_recvTime;
	std::map<uint32_t, std::vector<uint64_t>> m_delay;
	float avgSend;
	float avgRecv;
	float avgDelay;
	double avgThroughput;
	int loc;

	//packet loss rate
	void LostRate(uint32_t m_tuple,uint64_t rtt, uint64_t time);
	void IdTime(uint32_t m_tuple,uint64_t idtime);
	void ReTime(uint32_t m_tuple,uint64_t retime);
	void IdValue(uint32_t m_tuple,uint64_t idvalue);
	std::map<uint32_t, std::vector<uint64_t>>  idTime;
	std::map<uint32_t, std::vector<uint64_t>> reTime;
	std::map<uint32_t, std::vector<uint64_t>>  idValue;
	double lostRate;
*/
private:
	int m;

};

} // namespace ns3


#endif
