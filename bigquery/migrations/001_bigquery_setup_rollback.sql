-- Rollback script for BigQuery schema setup
-- Drops tables and dataset (use with caution!)

-- Drop tables first
DROP TABLE IF EXISTS `nvidia_blog.chunks`;
DROP TABLE IF EXISTS `nvidia_blog.items`;

-- Drop dataset (will fail if tables still exist)
DROP SCHEMA IF EXISTS `nvidia_blog`;

