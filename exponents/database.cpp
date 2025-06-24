#include <aws/core/Aws.h>
#include <aws/dynamodb/DynamoDBClient.h>
#include <aws/dynamodb/model/AttributeValue.h>
#include <aws/dynamodb/model/CreateTableRequest.h>
#include <aws/dynamodb/model/GetItemRequest.h>
#include <aws/dynamodb/model/PutItemRequest.h>
#include <iostream>

//! Create an Amazon DynamoDB table.
/*!
  \sa createTable()
  \param tableName: Name for the DynamoDB table.
  \param primaryKey: Primary key for the DynamoDB table.
  \param clientConfiguration: AWS client configuration.
  \return bool: Function succeeded.
 */
bool createTable(const Aws::String &tableName,
                 const Aws::String &primaryKey,
                 const Aws::Client::ClientConfiguration &clientConfiguration) {
    Aws::DynamoDB::DynamoDBClient dynamoClient(clientConfiguration);

    std::cout << "Creating table " << tableName <<
              " with a simple primary key: \"" << primaryKey << "\"." << std::endl;

    Aws::DynamoDB::Model::CreateTableRequest request;

    Aws::DynamoDB::Model::AttributeDefinition hashKey;
    hashKey.SetAttributeName(primaryKey);
    hashKey.SetAttributeType(Aws::DynamoDB::Model::ScalarAttributeType::S);
    request.AddAttributeDefinitions(hashKey);

    Aws::DynamoDB::Model::KeySchemaElement keySchemaElement;
    keySchemaElement.WithAttributeName(primaryKey).WithKeyType(
            Aws::DynamoDB::Model::KeyType::HASH);
    request.AddKeySchema(keySchemaElement);

    Aws::DynamoDB::Model::ProvisionedThroughput throughput;
    throughput.WithReadCapacityUnits(5).WithWriteCapacityUnits(5);
    request.SetProvisionedThroughput(throughput);
    request.SetTableName(tableName);

    const Aws::DynamoDB::Model::CreateTableOutcome &outcome = dynamoClient.CreateTable(
            request);
    if (outcome.IsSuccess()) {
        std::cout << "Table \""
                  << outcome.GetResult().GetTableDescription().GetTableName() <<
                  " created!" << std::endl;
    }
    else {
        std::cerr << "Failed to create table: " << outcome.GetError().GetMessage()
                  << std::endl;
        return false;
    }

    return outcome.IsSuccess();
}

// Modified putItem for your schema
bool putItem(const Aws::String& tableName,
             const Aws::String& date,
             double e0,
             double optimal_rho,
             int M,
             const Aws::String& const_type,
             double snr,
             double r,
             int n,
             const Aws::Client::ClientConfiguration& clientConfig) {
    Aws::DynamoDB::DynamoDBClient dynamoClient(clientConfig);
    Aws::DynamoDB::Model::PutItemRequest request;

    request.SetTableName(tableName);

    // Convert numbers to string representations
    auto numToString = [](double num) { return std::to_string(num); };

    request.AddItem("date", Aws::DynamoDB::Model::AttributeValue().SetS(date));
    request.AddItem("e0", Aws::DynamoDB::Model::AttributeValue().SetN(numToString(e0)));
    request.AddItem("optimal_rho", Aws::DynamoDB::Model::AttributeValue().SetN(numToString(optimal_rho)));
    request.AddItem("M", Aws::DynamoDB::Model::AttributeValue().SetN(std::to_string(M)));
    request.AddItem("constel", Aws::DynamoDB::Model::AttributeValue().SetS(const_type));
    request.AddItem("snr", Aws::DynamoDB::Model::AttributeValue().SetN(numToString(snr)));
    request.AddItem("r", Aws::DynamoDB::Model::AttributeValue().SetN(numToString(r)));
    request.AddItem("n", Aws::DynamoDB::Model::AttributeValue().SetN(numToString(n)));

    auto outcome = dynamoClient.PutItem(request);
    if (!outcome.IsSuccess()) {
        std::cerr << "PutItem error: " << outcome.GetError().GetMessage() << std::endl;
        return false;
    }
    return true;
}

struct ItemResult {
    double e0;
    double optimal_rho;
};

ItemResult getItem(
        const Aws::String &tableName,
        int M,
        const Aws::String &const_type,
        double snr,
        double r,
        int n,
        const Aws::Client::ClientConfiguration &clientConfig) {

    // Create DynamoDB Client
    Aws::DynamoDB::DynamoDBClient dynamoClient(clientConfig);

    // Build the key for GetItem request
    Aws::DynamoDB::Model::GetItemRequest request;
    request.SetTableName(tableName);

    Aws::Map<Aws::String, Aws::DynamoDB::Model::AttributeValue> key;

    // Convert numerical keys to string representation as DynamoDB stores them as numbers using SetN
    key["M"].SetN(std::to_string(M));
    key["const_type"].SetS(const_type);
    key["snr"].SetN(std::to_string(snr));
    key["r"].SetN(std::to_string(r));
    key["n"].SetN(std::to_string(n));

    request.SetKey(key);

    // Optionally, specify that you only want to retrieve e0 and optimal_rho to reduce data transfer
    Aws::Vector<Aws::String> attributesToGet{"e0", "optimal_rho"};
    request.SetAttributesToGet(attributesToGet);

    // Perform the GetItem operation
    auto outcome = dynamoClient.GetItem(request);

    if (!outcome.IsSuccess())
    {
        std::cerr << "Error getting item: "
                  << outcome.GetError().GetMessage() << std::endl;
        throw std::runtime_error("GetItem failed");
    }

    const auto& item = outcome.GetResult().GetItem();

    // Ensure the item was found
    if (item.empty())
    {
        std::cerr << "Item not found." << std::endl;
        throw std::runtime_error("Item not found");
    }

    ItemResult result;
    // Convert attributes from string to double
    if (item.find("e0") != item.end())
    {
        result.e0 = std::stod(item.at("e0").GetN());
    }
    else
    {
        std::cerr << "Attribute 'e0' not found." << std::endl;
        throw std::runtime_error("'e0' not found");
    }

    if (item.find("optimal_rho") != item.end())
    {
        result.optimal_rho = std::stod(item.at("optimal_rho").GetN());
    }
    else
    {
        std::cerr << "Attribute 'optimal_rho' not found." << std::endl;
        throw std::runtime_error("'optimal_rho' not found");
    }
    return result;
}



// Modified getItem for your schema
bool getItem(const Aws::String& tableName,
             const Aws::String& date,
             const Aws::Client::ClientConfiguration& clientConfig) {
    Aws::DynamoDB::DynamoDBClient dynamoClient(clientConfig);
    Aws::DynamoDB::Model::GetItemRequest request;

    request.SetTableName(tableName);
    request.AddKey("date", Aws::DynamoDB::Model::AttributeValue().SetS(date));

    auto outcome = dynamoClient.GetItem(request);
    if (outcome.IsSuccess()) {
        const auto& item = outcome.GetResult().GetItem();
        if (item.empty()) {
            std::cout << "Item not found!" << std::endl;
            return false;
        }

        std::cout << "Item contents:" << std::endl;
        for (const auto& attr : item) {
            if(attr.first != "constel" && attr.first != "date"){
                std::cout << "  " << attr.first << ": " << attr.second.GetN() << std::endl;
            }
            else{
                std::cout << "  " << attr.first << ": " << attr.second.GetS() << std::endl;
            }
        }
        return true;
    }

    std::cerr << "GetItem error: " << outcome.GetError().GetMessage() << std::endl;
    return false;
}