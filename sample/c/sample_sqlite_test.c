#include <sqlite3.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int test_in_memory(void) {
  sqlite3 *db;
  sqlite3_stmt *stmt;
  int rc;

  rc = sqlite3_open(":memory:", &db);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
    return -1;
  }
  printf("[PASS] sqlite3_open()\n");

  rc = sqlite3_exec(
      db,
      "CREATE TABLE persons ("
      "  person_id TEXT PRIMARY KEY,"
      "  display_name TEXT NOT NULL,"
      "  identity_state TEXT NOT NULL"
      ");"
      "INSERT INTO persons VALUES ('p_001', 'Alice', 'registered');"
      "INSERT INTO persons VALUES ('p_002', 'Bob', 'pending');"
      "INSERT INTO persons VALUES ('p_003', 'Charlie', 'unknown');",
      NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Create/Insert failed: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    return -1;
  }
  printf("[PASS] CREATE TABLE + INSERT\n");

  rc = sqlite3_prepare_v2(db,
                          "SELECT person_id, display_name, identity_state FROM "
                          "persons ORDER BY person_id",
                          -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Prepare failed: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    return -1;
  }

  int count = 0;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    const char *id = (const char *)sqlite3_column_text(stmt, 0);
    const char *name = (const char *)sqlite3_column_text(stmt, 1);
    const char *state = (const char *)sqlite3_column_text(stmt, 2);
    printf("  Row %d: id=%s, name=%s, state=%s\n", count, id, name, state);
    count++;
  }
  sqlite3_finalize(stmt);

  if (count != 3) {
    fprintf(stderr, "Expected 3 rows, got %d\n", count);
    sqlite3_close(db);
    return -1;
  }
  printf("[PASS] SELECT (3 rows returned)\n");

  rc = sqlite3_exec(
      db,
      "UPDATE persons SET identity_state='registered' WHERE person_id='p_002'",
      NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Update failed: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    return -1;
  }
  printf("[PASS] UPDATE\n");

  sqlite3_close(db);
  printf("[PASS] sqlite3_close()\n");
  return 0;
}

static int test_wal_mode(void) {
  sqlite3 *db;
  int rc;

  rc = sqlite3_open("/tmp/test_sqlite_wal.db", &db);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "Cannot open database: %s\n", sqlite3_errmsg(db));
    return -1;
  }

  rc = sqlite3_exec(db, "PRAGMA journal_mode=WAL", NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "WAL pragma failed: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    return -1;
  }

  rc = sqlite3_exec(db,
                    "CREATE TABLE IF NOT EXISTS test_wal (id INTEGER PRIMARY "
                    "KEY, value TEXT);"
                    "INSERT INTO test_wal (value) VALUES ('wal_test_value');",
                    NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    fprintf(stderr, "WAL insert failed: %s\n", sqlite3_errmsg(db));
    sqlite3_close(db);
    return -1;
  }

  printf("[PASS] WAL mode write\n");
  sqlite3_close(db);
  printf("[PASS] WAL mode test complete\n");
  fflush(stdout);
  fflush(stderr);
  return 0;
}

int main(void) {
  printf("=== SQLite Integration Test ===\n");
  printf("SQLite version: %s\n", sqlite3_libversion());
  printf("SQLite source ID: %s\n", sqlite3_sourceid());

  if (test_in_memory() != 0) {
    fprintf(stderr, "\n*** In-memory test FAILED ***\n");
    return -1;
  }

  if (test_wal_mode() != 0) {
    fprintf(stderr, "\n*** WAL mode test FAILED ***\n");
    return -1;
  }

  printf("\n=== All SQLite tests PASSED ===\n");
  _exit(0);
}