CREATE TABLE binlog_test(
	`id` int NOT NULL AUTO_INCREMENT,
    `payload` VARCHAR(200),
	`insert_time` int,
	`update_time` int,
    PRIMARY KEY (`id`)
);

INSERT INTO binlog_test VALUES (1, 'payload', 1, 1);
INSERT INTO binlog_test VALUES (2, 'payload', 2, 2);
INSERT INTO binlog_test VALUES (3, 'payload', 3, 3);
-- INSERT INTO binlog_test (`id`, `payload`, `insert_time`, `update_time`) VALUES (4, 'payload', 4, 4);
-- INSERT INTO binlog_test (`payload`) VALUES ('payload');

UPDATE binlog_test SET payload = 'payload2' WHERE id = 1;
UPDATE binlog_test SET payload = 'payload2' WHERE id = 2;
UPDATE binlog_test SET payload = 'payload2' WHERE id = 3;

DELETE FROM binlog_test WHERE id = 1;
DELETE FROM binlog_test WHERE id = 2;
DELETE FROM binlog_test WHERE id = 3;