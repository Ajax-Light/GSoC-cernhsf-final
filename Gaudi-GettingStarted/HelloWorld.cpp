#include "HelloWorld.h"

HelloWorld::HelloWorld("Hello World", ISvcLocator* pSvcLocator){
        // Declare the Algorithm Properties
        declareProperty("Int", m_int=100);
        declareProperty("Double", m_double=100.00);
        declarePropertly("String", m_string=std::string("One Hundred"));
    }

StatusCode HelloWorld::initialize(){
    MsgStream log(msgSvc(), name());
    log << MSG::INFO << "Initializing.....";
    log << MSG::INFO << "Property Int = " << m_int ;
    log << MSG::INFO << "Property Double = " << m_double ;
    log << MSG::INFO << "Property String = " << m_string ;

    //m_initialized = true;
    return StatusCode::SUCCESS;
}

StatusCode HelloWorld::execute(){
    MsgStream log( msgSvc(), name() );
    log << MSG::INFO << "Executing....";

    return StatusCode::SUCCESS;
}

StatusCode HelloWorld::finalize(){
    MsgStream log(msgSvc(), name());
    log << MSG::INFO << "Finalizing....";

    return StatusCode::SUCCESS;
}

DECLARE_COMPONENT(HelloWorld)