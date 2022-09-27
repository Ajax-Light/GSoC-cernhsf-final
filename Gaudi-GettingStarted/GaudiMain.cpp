// Standard Includes
#include <iostream>

// Gaudi Includes
#include "GaudiKernel/SmartIF.h"
#include "GaudiKernel/Bootstrap.h"
#include "GaudiKernel/IAppMgrUI.h"
#include "GaudiKernel/IProperty.h"


//--- Example main program
int main(int argc, char** argv) {

    // Create an instance of an application manager
    IInterface* iface = Gaudi::createApplicationMgr();

    SmartIF<IProperty> propMgr ( iface );
    SmartIF<IAppMgrUI> appMgr ( iface );

    if( !appMgr.isValid() || !propMgr.isValid() ) {
        std::cout << "Fatal error creating ApplicationMgr " << std::endl;
        return 1;
    }

    // Get the input configuration file from arguments
    std::string opts = (argc>1) ? argv[1] : "job.opts";

    propMgr->setProperty( "JobOptionsPath", opts );

    // Run the application manager and process events
    appMgr->run();

    // All done - exit
    return 0;
}
