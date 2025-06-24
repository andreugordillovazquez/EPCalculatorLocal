//
// Created by TESTER on 16/04/2025.
//

#include "database_main.h"
#include <iomanip>
#include "database.h"
#include <iostream>
#include <unordered_map>
#include <chrono>
#include <string>
#include <algorithm>
#include <fstream>
#include <locale>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>

int my_main(std::string tname) {
    Aws::Client::ClientConfiguration config;
    // (for example, set region: config.region = "us-west-2";)

    const Aws::String tableName(tname);

    try {
        // Example inputs that match your key attributes
        int M = 2;
        Aws::String const_type("PAM");
        double snr = 0.5;
        double r = 0;
        int n = 2;

        ItemResult res = getItem(tableName, M, const_type, snr, r, n, config);
        std::cout << "e0: " << res.e0 << "\n";
        std::cout << "optimal_rho: " << res.optimal_rho << "\n";
    } catch (const std::exception &ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
    }

    return 0;
}