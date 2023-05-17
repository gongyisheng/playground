CREATE TABLE 1kb_test (
    id INT NOT NULL AUTO_INCREMENT,
    payload VARCHAR(1000),
    PRIMARY KEY (id)
);

CREATE TABLE 1mb_test (
    id INT NOT NULL AUTO_INCREMENT,
    payload VARCHAR(1000000),
    PRIMARY KEY (id)
);

CREATE TABLE 100k_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 200k_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 500k_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 1m_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 2m_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 5m_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 10m_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 20m_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);

CREATE TABLE 20m_row_test(
	`id` int NOT NULL AUTO_INCREMENT,
	`person_id` int NOT NULL,
	`person_name` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`),
    KEY `query_by_update_time` (`update_time`),
    KEY `query_by_person_id_insert_time` (`insert_time`)
);