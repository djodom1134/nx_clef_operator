#include "http_client.h"

#include <cstring>

#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

namespace clef_operator {

namespace {

struct Url
{
    std::string host;
    std::string port = "80";
    std::string path = "/";
};

bool parseUrl(const std::string& url, Url* out, std::string* error)
{
    const std::string scheme = "http://";
    if (url.compare(0, scheme.size(), scheme) != 0)
    {
        *error = "only http:// URLs are supported: " + url;
        return false;
    }
    const std::string rest = url.substr(scheme.size());
    const auto slash = rest.find('/');
    const std::string hostPort = rest.substr(0, slash);
    if (slash != std::string::npos)
        out->path = rest.substr(slash);
    const auto colon = hostPort.rfind(':');
    if (colon != std::string::npos)
    {
        out->host = hostPort.substr(0, colon);
        out->port = hostPort.substr(colon + 1);
    }
    else
    {
        out->host = hostPort;
    }
    if (out->host.empty())
    {
        *error = "no host in URL: " + url;
        return false;
    }
    return true;
}

HttpResponse request(
    const std::string& method,
    const std::string& url,
    const std::string& body,
    const std::string& bearerToken,
    int timeoutSeconds)
{
    HttpResponse response;
    Url parsed;
    if (!parseUrl(url, &parsed, &response.error))
        return response;

    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* addresses = nullptr;
    if (const int rc = getaddrinfo(parsed.host.c_str(), parsed.port.c_str(), &hints, &addresses); rc != 0)
    {
        response.error = std::string("cannot resolve ") + parsed.host + ": " + gai_strerror(rc);
        return response;
    }

    int fd = -1;
    for (addrinfo* a = addresses; a; a = a->ai_next)
    {
        fd = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (fd < 0)
            continue;
        timeval timeout{timeoutSeconds, 0};
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        if (connect(fd, a->ai_addr, a->ai_addrlen) == 0)
            break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(addresses);
    if (fd < 0)
    {
        response.error = "cannot connect to " + parsed.host + ":" + parsed.port;
        return response;
    }

    std::string head = method + " " + parsed.path + " HTTP/1.1\r\n"
        "Host: " + parsed.host + "\r\n"
        "Connection: close\r\n";
    if (!bearerToken.empty())
        head += "Authorization: Bearer " + bearerToken + "\r\n";
    if (method == "POST")
    {
        head += "Content-Type: application/json\r\n"
            "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    head += "\r\n";
    const std::string message = head + body;

    size_t sent = 0;
    while (sent < message.size())
    {
        const ssize_t n = send(fd, message.data() + sent, message.size() - sent, MSG_NOSIGNAL);
        if (n <= 0)
        {
            response.error = "send failed";
            close(fd);
            return response;
        }
        sent += (size_t) n;
    }

    std::string raw;
    char buffer[16384];
    for (;;)
    {
        const ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
        if (n < 0)
        {
            response.error = "receive failed or timed out";
            close(fd);
            return response;
        }
        if (n == 0)
            break;
        raw.append(buffer, (size_t) n);
    }
    close(fd);

    const auto headerEnd = raw.find("\r\n\r\n");
    if (headerEnd == std::string::npos || raw.compare(0, 5, "HTTP/") != 0)
    {
        response.error = "malformed HTTP response";
        return response;
    }
    const auto space = raw.find(' ');
    response.status = std::atoi(raw.c_str() + space + 1);
    response.body = raw.substr(headerEnd + 4);
    return response;
}

} // namespace

HttpResponse httpPostJson(
    const std::string& url, const std::string& body, const std::string& bearerToken, int timeoutSeconds)
{
    return request("POST", url, body, bearerToken, timeoutSeconds);
}

HttpResponse httpGet(const std::string& url, const std::string& bearerToken, int timeoutSeconds)
{
    return request("GET", url, "", bearerToken, timeoutSeconds);
}

std::string base64Encode(const std::string& data)
{
    static const char* kAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((data.size() + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < data.size(); i += 3)
    {
        const uint32_t v = ((uint8_t) data[i] << 16) | ((uint8_t) data[i + 1] << 8) | (uint8_t) data[i + 2];
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += kAlphabet[(v >> 6) & 63];
        out += kAlphabet[v & 63];
    }
    if (i < data.size())
    {
        uint32_t v = (uint8_t) data[i] << 16;
        if (i + 1 < data.size())
            v |= (uint8_t) data[i + 1] << 8;
        out += kAlphabet[(v >> 18) & 63];
        out += kAlphabet[(v >> 12) & 63];
        out += (i + 1 < data.size()) ? kAlphabet[(v >> 6) & 63] : '=';
        out += '=';
    }
    return out;
}

} // namespace clef_operator
