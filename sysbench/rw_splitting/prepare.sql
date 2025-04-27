CREATE DATABASE IF NOT EXISTS rw_splitting_test;
USE rw_splitting_test;

CREATE TABLE test (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    number1 INT UNSIGNED,
    number2 INT UNSIGNED,
    number3 INT UNSIGNED,
    payload1 VARCHAR(255),
    payload2 VARCHAR(255),
    payload3 VARCHAR(255),
    KEY idx_unique (number1, number2, number3)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;