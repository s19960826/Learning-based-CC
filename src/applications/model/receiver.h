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

#ifndef RECEIVER_H
#define RECEIVER_H

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/application.h"

#include "ns3/stats-module.h"

using namespace ns3;

//------------------------------------------------------
class Receiver : public Application {
public:
  static TypeId GetTypeId (void);
  Receiver();
  virtual ~Receiver();

  void SetCounter (Ptr<CounterCalculator<> > calc);

  //send timestamp, ack timestamp and delay

  Ptr<TimeMinMaxAvgTotalCalculator> delay_new;
  Ptr<TimeMinMaxAvgTotalCalculator> send_new;
  Ptr<TimeMinMaxAvgTotalCalculator> recv_new;
  //Ptr<TimeMinMaxAvgTotalCalculator> m_delay;
  std::vector<Ptr<TimeMinMaxAvgTotalCalculator> > time_send;
  std::vector<Ptr<TimeMinMaxAvgTotalCalculator> > time_recv;
  std::vector<Ptr<TimeMinMaxAvgTotalCalculator> > m_delay;

protected:
  virtual void DoDispose (void);

private:
  virtual void StartApplication (void);
  virtual void StopApplication (void);

  void Receive (Ptr<Socket> socket);

  Ptr<Socket>     m_socket;

  uint32_t        m_port;

  Ptr<CounterCalculator<> > m_calc;

  // end class Receiver
};

#endif /* RECEIVER_H */
