// Include files
#include "GaudiKernel/Algorithm.h"
// Required for inheritance
// #include "GaudiKernel/Property.h"
#include "GaudiKernel/MsgStream.h"

class HelloWorld : public Algorithm {

public:

  /// Constructor of this form must be provided
  HelloWorld(const std::string& name, ISvcLocator* pSvcLocator);

  /// Three mandatory member functions of any algorithm
  StatusCode initialize();
  StatusCode execute();
  StatusCode finalize();

private:

  /// These data members are used in the execution of this algorithm
  /// and are set in the initialisation phase by the job options service
  int m_int;
  double m_double;
  std::string m_string;
};
