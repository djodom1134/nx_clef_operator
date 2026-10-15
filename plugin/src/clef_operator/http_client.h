#pragma once

#include <string>

namespace clef_operator {

struct HttpResponse
{
    int status = 0; //< 0 when the request failed before a response arrived.
    std::string body;
    std::string error;
};

/** Blocking HTTP/1.1 POST of a JSON body to an http:// URL. No TLS: the gateway is on the LAN. */
HttpResponse httpPostJson(
    const std::string& url,
    const std::string& body,
    const std::string& bearerToken,
    int timeoutSeconds);

/** Blocking HTTP/1.1 GET. */
HttpResponse httpGet(const std::string& url, const std::string& bearerToken, int timeoutSeconds);

std::string base64Encode(const std::string& data);

} // namespace clef_operator
