#include "clef_operator/integration.h"

extern "C" NX_PLUGIN_API nx::sdk::IIntegration* createNxPlugin()
{
    return new clef_operator::Integration();
}
