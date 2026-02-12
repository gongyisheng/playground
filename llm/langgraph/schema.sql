
/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
DROP TABLE IF EXISTS `checkpoints`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `checkpoints` (
  `thread_id` varchar(150) NOT NULL,
  `checkpoint_ns` varchar(2000) NOT NULL DEFAULT '',
  `checkpoint_id` varchar(150) NOT NULL,
  `parent_checkpoint_id` varchar(150) DEFAULT NULL,
  `type` varchar(150) DEFAULT NULL,
  `checkpoint` json NOT NULL,
  `metadata` json NOT NULL DEFAULT (_utf8mb4'{}'),
  `checkpoint_ns_hash` binary(16) NOT NULL,
  PRIMARY KEY (`thread_id`,`checkpoint_ns_hash`,`checkpoint_id`),
  KEY `checkpoints_thread_id_idx` (`thread_id`),
  KEY `checkpoints_checkpoint_id_idx` (`checkpoint_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `checkpoint_blobs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `checkpoint_blobs` (
  `thread_id` varchar(150) NOT NULL,
  `checkpoint_ns` varchar(2000) NOT NULL DEFAULT '',
  `channel` varchar(150) NOT NULL,
  `version` varchar(150) NOT NULL,
  `type` varchar(150) NOT NULL,
  `blob` longblob,
  `checkpoint_ns_hash` binary(16) NOT NULL,
  PRIMARY KEY (`thread_id`,`checkpoint_ns_hash`,`channel`,`version`),
  KEY `checkpoint_blobs_thread_id_idx` (`thread_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `checkpoint_writes`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `checkpoint_writes` (
  `thread_id` varchar(150) NOT NULL,
  `checkpoint_ns` varchar(2000) NOT NULL DEFAULT '',
  `checkpoint_id` varchar(150) NOT NULL,
  `task_id` varchar(150) NOT NULL,
  `idx` int NOT NULL,
  `channel` varchar(150) NOT NULL,
  `type` varchar(150) DEFAULT NULL,
  `blob` longblob NOT NULL,
  `checkpoint_ns_hash` binary(16) NOT NULL,
  `task_path` varchar(2000) NOT NULL DEFAULT '',
  PRIMARY KEY (`thread_id`,`checkpoint_ns_hash`,`checkpoint_id`,`task_id`,`idx`),
  KEY `checkpoint_writes_thread_id_idx` (`thread_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `checkpoint_migrations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `checkpoint_migrations` (
  `v` int NOT NULL,
  PRIMARY KEY (`v`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

