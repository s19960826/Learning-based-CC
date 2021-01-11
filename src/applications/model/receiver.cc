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

#include "ns3/receiver.h"
#include "ns3/time-stamp-tag.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("Receiver");

NS_OBJECT_ENSURE_REGISTERED (Receiver);

//----------------------------------------------------------------------
//-- Receiver
//------------------------------------------------------
TypeId
Receiver::GetTypeId (void)
{
  static TypeId tid = TypeId ("Receiver")
    .SetParent<Application> ()
    .SetGroupName ("Applications")
    .AddConstructor<Receiver> ()
    .AddAttribute ("Port", "Listening port.",
                   UintegerValue (1603),
                   MakeUintegerAccessor (&Receiver::m_port),
                   MakeUintegerChecker<uint32_t>())
  ;
  return tid;
}

Receiver::Receiver() :
  m_calc (0),
  delay_new(0)
{
  NS_LOG_FUNCTION_NOARGS ();
  m_socket = 0;
}

Receiver::~Receiver()
{
  NS_LOG_FUNCTION_NOARGS ();
}

void
Receiver::DoDispose (void)
{
  NS_LOG_FUNCTION_NOARGS ();

  m_socket = 0;
  // chain up
  Application::DoDispose ();
}

void
Receiver::StartApplication ()
{
  NS_LOG_FUNCTION_NOARGS ();

  if (m_socket == 0) {
      Ptr<SocketFactory> socketFactory = GetNode ()->GetObject<SocketFactory>
          (UdpSocketFactory::GetTypeId ());
      m_socket = socketFactory->CreateSocket ();
      InetSocketAddress local = 
        InetSocketAddress (Ipv4Address::GetAny (), m_port);
      m_socket->Bind (local);
    }

  m_socket->SetRecvCallback (MakeCallback (&Receiver::Receive, this));
  std::cout << "this works" << std::endl;
  // end Receiver::StartApplication
}

void
Receiver::StopApplication ()
{
  NS_LOG_FUNCTION_NOARGS ();

  if (m_socket != 0) {
      m_socket->SetRecvCallback (MakeNullCallback<void, Ptr<Socket> > ());
    }

  // end Receiver::StopApplication
}

void
Receiver::SetCounter (Ptr<CounterCalculator<> > calc)
{
  m_calc = calc;
  // end Receiver::SetCounter
}


void
Receiver::Receive (Ptr<Socket> socket)
{
  // NS_LOG_FUNCTION (this << socket << packet << from);

  Ptr<Packet> packet;
  Address from;
  while ((packet = socket->RecvFrom (from))) {
      if (InetSocketAddress::IsMatchingType (from)) {
          NS_LOG_INFO ("Received " << packet->GetSize () << " bytes from " <<
                       InetSocketAddress::ConvertFrom (from).GetIpv4 ());
        }
      TimestampTag timestamp;
      // Should never not be found since the sender is adding it, but
      // you never know.
      if (packet->FindFirstMatchingByteTag (timestamp)) {
          Time tx = timestamp.GetTimestamp ();
          
         /* 
          //get send timestamp, ack timestamp and delay
          Time now = Simulator::Now();
          delay_new->Update(now-tx);
          send_new->Update(tx);
          recv_new->Update(now);
          m_delay.push_back(delay_new);
          time_send.push_back(send_new);
          time_recv.push_back(recv_new);
	  std::cout << "this works" << std::endl;
        */
       }

      if (m_calc != 0) {
          m_calc->Update ();
        }

      // end receiving packets
    }

  // end Receiver::Receive
}

