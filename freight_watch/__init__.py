try:
    import truststore

    # Trust the OS certificate store so HTTPS works behind TLS-inspecting corporate proxies.
    truststore.inject_into_ssl()
except ImportError:
    pass
