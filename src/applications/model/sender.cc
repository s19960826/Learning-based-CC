/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Authors: Joe Kopena <tjkopena@cs.drexel.edu>
 *
 * These applications are used in the WiFi Distance Test experiment,
 * described and implemented in test02.cc.  That file should be in the
 * same place as this file.  The applications have two very simple
 * jobs, they just generate and receive packets.  We could use the
 * standard Application classes included in the NS-3 distribution.
 * These have been written just to change the behavior a little, and
 * provide more examples.
 *
 */

#include <ostream>

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"

#include "ns3/stats-module.h"

#include "ns3/sender.h"
#include "ns3/time-stamp-tag.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Sender");

NS_OBJECT_ENSURE_REGISTERED (Sender);

TypeId
Sender::GetTypeId (void)
{
  static TypeId tid = TypeId ("Sender")
    .SetParent<Application> ()
    .SetGroupName ("Applications")
    .AddConstructor<Sender> ()
    .AddAttribute ("PacketSize", "The size of packets transmitted.",
                   UintegerValue (64),
                   MakeUintegerAccessor (&Sender::m_pktSize),
                   MakeUintegerChecker<uint32_t>(1))
    .AddAttribute ("Destination", "Target host address.",
                   Ipv4AddressValue ("255.255.255.255"),
                   MakeIpv4AddressAccessor (&Sender::m_destAddr),
                   MakeIpv4AddressChecker ())
    .AddAttribute ("Port", "Destination app port.",
                   UintegerValue (1603),
                   MakeUintegerAccessor (&Sender::m_destPort),
                   MakeUintegerChecker<uint32_t>())
    .AddAttribute ("NumPackets", "Total number of packets to send.",
                   UintegerValue (30),
                   MakeUintegerAccessor (&Sender::m_numPkts),
                   MakeUintegerChecker<uint32_t>(1))
    .AddAttribute ("Interval", "Delay between transmissions.",
                   StringValue ("ns3::ConstantRandomVariable[Constant=0.5]"),
                   MakePointerAccessor (&Sender::m_interval),
                   MakePointerChecker <RandomVariableStream>())
    .AddTraceSource ("Tx", "A new packet is created and is sent",
                     MakeTraceSourceAccessor (&Sender::m_txTrace),
                     "ns3::Packet::TracedCallback")
  ;
  return tid;
}

Sender::Sender()
{
  NS_LOG_FUNCTION_NOARGS ();
  m_interval = CreateObject<ConstantRandomVariable> ();
  m_socket = 0;
}

Sender::~Sender()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
Sender::Setup (Ipv4Address destAddr)
{
  m_destAddr = destAddr;
}

void
Sender::DoDispose (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  m_socket = 0;
  // chain up
  Application::DoDispose ();
}

void Sender::StartApplication ()
{
  NS_LOG_FUNCTION_NOARGS ();

  if (m_socket == 0) {
      Ptr<SocketFactory> socketFactory = GetNode ()->GetObject<SocketFactory>
          (UdpSocketFactory::GetTypeId ());
      m_socket = socketFactory->CreateSocket ();
      m_socket->Bind ();
    }

  m_count = 0;

  Simulator::Cancel (m_sendEvent);
  m_sendEvent = Simulator::ScheduleNow (&Sender::SendPacket, this);

  // end Sender::StartApplication
}

void Sender::StopApplication ()
{
  NS_LOG_FUNCTION_NOARGS ();
  Simulator::Cancel (m_sendEvent);
  // end Sender::StopApplication
}

void Sender::SendPacket ()
{
  // NS_LOG_FUNCTION_NOARGS ();
  NS_LOG_INFO ("Sending packet at " << Simulator::Now () << " to " <<
               m_destAddr);
  
  Ptr<Packet> packet = Create<Packet>(m_pktSize); 

  //cerate timestamptag
  TimestampTag timestamp;
  timestamp.SetTimestamp (Simulator::Now ());
  packet->AddByteTag (timestamp);

  // Could connect the socket since the address never changes; using SendTo
  // here simply because all of the standard apps do not.
  m_socket->SendTo (packet, 0, InetSocketAddress (m_destAddr, m_destPort));

  // Report the event to the trace.
  m_txTrace (packet);

  if (++m_count < m_numPkts) {
      m_sendEvent = Simulator::Schedule (Seconds (m_interval->GetValue ()),
                                         &Sender::SendPacket, this);
    }

  // end Sender::SendPacket
}
