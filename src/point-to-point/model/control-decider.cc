#include "control-decider.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include <numeric>
#include <fstream>
#include <sstream>
#include <math.h>


namespace ns3
{

NS_LOG_COMPONENT_DEFINE ("ControlDecider");

NS_OBJECT_ENSURE_REGISTERED (ControlDecider);

TypeId
ControlDecider::GetTypeId (void)
{
  	static TypeId tid = TypeId ("ns3::ControlDecider")
	    .SetParent<Object> ()
	    .SetGroupName("PointToPoint")
	    .AddConstructor<ControlDecider> ()
	    ;
	return tid;
}


ControlDecider::ControlDecider()
{
	NS_LOG_FUNCTION(this);
}

ControlDecider::~ControlDecider()
{
	NS_LOG_FUNCTION(this);
}

void
ControlDecider::mapTuple(uint32_t m_tuple, uint64_t packetSize)
{
	std::map<uint32_t, std::vector<uint64_t>>::iterator iter;
	iter = tmp_mapTuple.find(m_tuple);
  
	if(iter != tmp_mapTuple.end())
	{
		std::vector<uint64_t> tmp = iter->second;
		if(tmp.size()==2)
		{
			tmp.push_back(packetSize);
			tmp_mapTuple.erase(m_tuple);
			tmp_mapTuple.insert(std::pair<uint32_t, std::vector<uint64_t>>(m_tuple, tmp));
      ControlDecider::finalTuple(m_tuple);
		}
		if(tmp.size()==1)
		{
			tmp.push_back(packetSize);
			tmp_mapTuple.erase(m_tuple);
			tmp_mapTuple.insert(std::pair<uint32_t, std::vector<uint64_t>>(m_tuple, tmp));
		}
	}
	else
	{
		std::vector<uint64_t> tmp;
		tmp.push_back(packetSize);
		tmp_mapTuple.insert(std::pair<uint32_t, std::vector<uint64_t>>(m_tuple, tmp));
	}
  
}

void
ControlDecider::classifyFlows(uint32_t m_tuple)
{
  srand(randtmp);
  uint64_t ranf = rand()%100;
  randtmp += 1;
  m_type.insert(std::pair<uint32_t, std::vector<uint64_t>>(m_tuple, ranf));
}


void
ControlDecider::finalTuple(uint32_t m_tuple)
{
  std::map<uint32_t, std::vector<uint64_t>>::iterator iter;
	iter = tmp_mapTuple.find(m_tuple);
  if(iter != tmp_mapTuple.end())
  {
    std::vector<uint64_t> tmp = iter->second;
    final_mapTuple.insert(std::pair<uint32_t, std::vector<uint64_t>>(m_tuple, tmp));
  }
}


void
ControlDecider::SendTime(uint32_t m_tuple, uint64_t sendValue)
{
  std::vector<uint64_t> tmp = m_sendTime[m_tuple];
  tmp.push_back(sendValue);
  m_sendTime[m_tuple] = tmp;
}


void
ControlDecider::RecvTime(uint32_t m_tuple,uint64_t recvValue)
{
  std::vector<uint64_t> tmp = m_recvTime[m_tuple];
  tmp.push_back(recvValue);
  m_recvTime[m_tuple] = tmp;
}

void
ControlDecider::IdTime(uint32_t m_tuple,uint64_t  idtime)
{
  std::vector<uint64_t> tmp = idTime[m_tuple];
  tmp.push_back(idtime);
  idTime[m_tuple] = tmp;
}


void
ControlDecider::ReTime(uint32_t m_tuple,uint64_t retime)
{
  std::vector<uint64_t> tmp = reTime[m_tuple];
  tmp.push_back(retime);
  reTime[m_tuple] = tmp;
}

void
ControlDecider::IdValue(uint32_t m_tuple,uint64_t idvalue)
{
  std::vector<uint64_t> tmp = idValue[m_tuple];
  tmp.push_back(idvalue);
  idValue[m_tuple] = tmp;
}

double
ControlDecider::AvgThroughput(uint32_t m_tuple,uint64_t rtt, uint64_t time)
{
  std::vector<uint64_t> m_sendTimes = m_sendTime[m_tuple];
  std::vector<uint64_t> m_recvTimes = m_recvTime[m_tuple];
  int timeSize=m_sendTimes.size();
  int count = 0;

  for(int i=timeSize-1; time-m_recvTimes[i]<=rtt and i>=0; i--)
  {
    count = count+1;
  }
  avgThroughput=pow(10,6)*8*340*count/float(time-m_recvTimes[count-1]);
  return(avgThroughput);  
}


double
ControlDecider::LostRate(uint32_t m_tuple,uint64_t rtt, uint64_t time)
{
  std::vector<uint64_t> m_idTime=idTime[m_tuple];
  std::vector<uint64_t> m_reTime=reTime[m_tuple];
  int idSize=m_idTime.size();
  int reSize=m_reTime.size();
  int idPack=0;
  int rePack=0;

  if(idSize>0 and reSize>0)
  {
    for(int i=idSize-1; time-m_idTime[i]<=rtt and i>=0;i--)
    {
      idPack = idPack+1;
    }

    for(int i=reSize-1; time-m_reTime[i]<=rtt and i>=0;i--)
    {
      rePack = rePack+1;
    }
    lostRate = pow(10,6)*rePack/double(idPack);
    return(lostRate);
  }
  else
  {
    return(0.0);
  }
}

double
ControlDecider::AvgRTT(uint32_t m_tuple, std::vector<uint64_t> m_rtt, std::vector<uint64_t> m_rtttime, uint64_t rtt, uint64_t time)
{

  int timeSize=m_rtt.size();
  int sumRtt = 0;
  int count = 0;
  for(int i=timeSize-1; time-m_rtttime[i]<=rtt and i>=0; i--)
  {
    sumRtt = sumRtt+m_rtt[i];
    count = i;
  }
  avgRtt = sumRtt/double(count);
  return(avgRtt);  
}


} // namespace ns3





