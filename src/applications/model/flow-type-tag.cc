#include <ostream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"

#include "ns3/stats-module.h"

#include "ns3/flow-type-tag.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("FlowTypeTag");

NS_OBJECT_ENSURE_REGISTERED (FlowTypeTag);


//----------------------------------------------------------------------
//-- FlowTypeTag
//------------------------------------------------------
TypeId 
FlowTypeTag::GetTypeId (void)
{
  static TypeId tid = TypeId ("FlowTypeTag")
    .SetParent<Tag> ()
    .SetGroupName ("Appilications")
    .AddConstructor<FlowTypeTag> ()
    .AddAttribute ("FlowType",
                   "the type of flow size!",
                   EmptyAttributeValue (),
                   MakeUintegerAccessor (&FlowTypeTag::GetFlowType),
                   MakeUintegerChecker<uint64_t> (1))
  ;
  return tid;
}
TypeId 
FlowTypeTag::GetInstanceTypeId (void) const
{
  return GetTypeId ();
}

uint32_t 
FlowTypeTag::GetSerializedSize (void) const
{
  return 8;
}

void 
FlowTypeTag::Serialize (TagBuffer i) const
{
  //int64_t t = m_type;
  //i.Write ((const uint8_t *)&t, 8);
  i.WriteU16(m_type);
}

void 
FlowTypeTag::Deserialize (TagBuffer i)
{
  //int64_t t;
  //i.Read ((uint8_t *)&t, 8);
  //m_type = t;
  m_type = i.ReadU16();
}

void
FlowTypeTag::SetFlowType (uint64_t type)
{
  m_type = type;
}

uint64_t
FlowTypeTag::GetFlowType (void) const
{
  return m_type;
}

void 
FlowTypeTag::Print (std::ostream &os) const
{
  os << "t=" << m_type;
}
