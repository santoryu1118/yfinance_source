##!/bin/bash
#
## run by `sh register_connector.sh connector.json`
#
## Register the connector
#echo "Registering connector..."
#curl -X POST --location "http://localhost:8083/connectors" \
#    -H "Content-Type: application/json" \
#    -H "Accept: application/json" \
#    -d @debezium/$1
#
## Check the exit status of the last command (curl)
#if [ $? -eq 0 ]; then
#    echo "Connector successfully registered"
#else
#    echo "Connector register failed"
#fi
#


#!/bin/bash

# run by `sh register_connector.sh connector.json`

# Register the connector
echo "Registering connector..."
response=$(curl -s -o response.txt -w "%{http_code}" --location --request POST "http://localhost:8083/connectors" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d @debezium/$1)

# Check HTTP status code
if [ "$response" -ge 200 ] && [ "$response" -lt 300 ]; then
    echo "Connector successfully registered"
else
    echo "Connector register failed"
    cat response.txt  # Outputs the response body for debugging
fi

# Clean up the temporary response file
rm response.txt
