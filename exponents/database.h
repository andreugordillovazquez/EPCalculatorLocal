//
// Created by TESTER on 14/04/2025.
//
#include <aws/core/Aws.h>

#ifndef UNTITLED12_DATABASE_H
#define UNTITLED12_DATABASE_H


bool getItem(const Aws::String& tableName,
             const Aws::String& date,
             const Aws::Client::ClientConfiguration& clientConfig);

bool putItem(const Aws::String& tableName,
             const Aws::String& date,
             double e0,
             double optimal_rho,
             int M,
             const Aws::String& const_type,
             double snr,
             double r,
             int n,
             const Aws::Client::ClientConfiguration& clientConfig);
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
        const Aws::Client::ClientConfiguration &clientConfig);

bool createTable(const Aws::String &tableName,
                 const Aws::String &primaryKey,
                 const Aws::Client::ClientConfiguration &clientConfiguration);




#endif //UNTITLED12_DATABASE_H
