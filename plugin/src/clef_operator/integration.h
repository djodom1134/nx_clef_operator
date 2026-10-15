#pragma once

#include <nx/sdk/analytics/helpers/integration.h>
#include <nx/sdk/analytics/i_engine.h>

namespace clef_operator {

class Integration: public nx::sdk::analytics::Integration
{
protected:
    virtual nx::sdk::Result<nx::sdk::analytics::IEngine*> doObtainEngine() override;
    virtual std::string manifestString() const override;
};

} // namespace clef_operator
