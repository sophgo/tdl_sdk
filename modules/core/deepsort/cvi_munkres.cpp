/*
 * reference:
 *     https://www.feiyilin.com/munkres.html
 */

#include "cvi_munkres.hpp"

#include <iomanip>
#include <iostream>

#define DEBUG 0

CVIMunkres::CVIMunkres(Eigen::MatrixXf *matrix) {
  original_matrix = matrix;
  rows = matrix->rows();
  cols = matrix->cols();
  m_size = MAX(rows, cols);

  covered_C = new bool[m_size];
  covered_R = new bool[m_size];
  prime_index[0] = -1;
  prime_index[1] = -1;
  match_result = nullptr;

  /* Initialization */
  state_matrix = new int *[m_size];
  cost_matrix = Eigen::MatrixXf(m_size, m_size);

  for (int i = 0; i < m_size; i++) {
    state_matrix[i] = new int[m_size];
    for (int j = 0; j < m_size; j++) {
      state_matrix[i][j] = NODE_STATE::NONE;
      if (i < rows && j < cols) {
        cost_matrix(i, j) = (*matrix)(i, j);
      } else {
        cost_matrix(i, j) = PADDING_VALUE;
      }
    }

    covered_R[i] = false;
    covered_C[i] = false;
  }

#if DEBUG
  std::cout << "CVIMunkres::cost_matrix" << std::endl;
  std::cout << cost_matrix << std::endl;
  std::cout << "CVIMunkres::original_matrix" << std::endl;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      std::cout << original_matrix[i][j] << ", ";
    }
    std::cout << std::endl;
  }
#endif /* DEBUG */
}

int CVIMunkres::solve() {
  int stage = ALGO_STAGE::ZERO;
#if DEBUG
  int stage_counter = 0;
#endif /* DEBUG */
  while (true) {
#if DEBUG
    stage_counter += 1;
    std::cout << "stage_counter = " << stage_counter << std::endl;
    if (stage_counter > 1000) {
      assert(0);
    }
#endif /* DEBUG */
    switch (stage) {
      case ALGO_STAGE::ZERO:
        stage = stage_0();
        break;
      case ALGO_STAGE::ONE:
        stage = stage_1();
        break;
      case ALGO_STAGE::TWO:
        stage = stage_2();
        break;
      case ALGO_STAGE::THREE:
        stage = stage_3();
        break;
      case ALGO_STAGE::FOUR:
        stage = stage_4();
        break;
      case ALGO_STAGE::FINAL:
        stage = stage_final();
        break;
      case ALGO_STAGE::DONE:
        stage = stage_final();
        return 0;
      default:
        assert(0);
    }
  }

  return 1;
}

int CVIMunkres::stage_0() {
#if DEBUG
  std::cout << "stage 0" << std::endl;
#endif /* DEBUG */
  if (rows > cols) {
    substract_min_by_col();
  } else if (rows < cols) {
    substract_min_by_row();
  } else { /* rows == cols */
    substract_min_by_col();
    substract_min_by_row();
  }
#if DEBUG
  std::cout << "cost_matrix" << std::endl;
  std::cout << cost_matrix << std::endl;
#endif /* DEBUG */

  for (int i = 0; i < m_size; i++) {
    for (int j = 0; j < m_size; j++) {
      if (cost_matrix(i, j) == 0 && !covered_R[i] && !covered_C[j]) {
        state_matrix[i][j] = NODE_STATE::STAR;
        covered_R[i] = true;
        covered_C[j] = true;
        break;
      }
    }
  }

  return ALGO_STAGE::ONE;
}

int CVIMunkres::stage_1() {
#if DEBUG
  std::cout << "stage 1" << std::endl;
#endif /* DEBUG */
  for (int i = 0; i < m_size; i++) {
    covered_R[i] = false;
    covered_C[i] = false;
  }

  int star_counter = 0;
  for (int i = 0; i < m_size; i++) {
    for (int j = 0; j < m_size; j++) {
      if (state_matrix[i][j] == NODE_STATE::PRIME) {
        state_matrix[i][j] = NODE_STATE::NONE;
      } else if (state_matrix[i][j] == NODE_STATE::STAR) {
        star_counter += 1;
        covered_C[j] = true;
      }
    }
  }

  assert(star_counter <= m_size);
  if (star_counter == m_size) {
    return ALGO_STAGE::FINAL;
  } else {
    return ALGO_STAGE::TWO;
  }
}

int CVIMunkres::stage_2() {
#if DEBUG
  std::cout << "stage 2" << std::endl;
  show_state_matrix();
#endif /* DEBUG */
  int _counter = 0;
  while (true) {
    int i, j;
    if (!find_uncovered_zero(i, j)) {
      return ALGO_STAGE::FOUR;
    }

    state_matrix[i][j] = NODE_STATE::PRIME;
#if DEBUG
    show_state_matrix();
#endif /* DEBUG */
    if (find_star_by_row(i, j)) {
      covered_R[i] = true;
      covered_C[j] = false;
    } else {
      prime_index[0] = i;
      prime_index[1] = j;
      return ALGO_STAGE::THREE;
    }

    _counter += 1;
    if (_counter > 100) {
      std::cout << "too many times in while loop...\n";
      exit(-1);
    }
  }
}

int CVIMunkres::stage_3() {
#if DEBUG
  std::cout << "stage 3" << std::endl;
  show_state_matrix();
#endif /* DEBUG */
  int **node_sequence = new int *[2 * m_size + 1];
  node_sequence[0] = new int[2];
  node_sequence[0][0] = prime_index[0];
  node_sequence[0][1] = prime_index[1];
  int node_counter = 1;

  int r = -1, c = prime_index[1];
#if DEBUG
  std::cout << r << ", " << c << std::endl;
#endif /* DEBUG */
  while (find_star_by_col(c, r)) {
    node_sequence[node_counter] = new int[2];
    node_sequence[node_counter][0] = r;
    node_sequence[node_counter][1] = c;
    node_counter += 1;

    if (!find_prime_by_row(r, c)) {
      std::cout << "ERROR: Prime Node not found.\n";
      exit(-1);
    }

    node_sequence[node_counter] = new int[2];
    node_sequence[node_counter][0] = r;
    node_sequence[node_counter][1] = c;
    node_counter += 1;
  }

  for (int i = 0; i < node_counter; i++) {
    r = node_sequence[i][0];
    c = node_sequence[i][1];
#if DEBUG
    std::cout << "(" << i << ") [" << std::setw(2) << r << "," << std::setw(2) << c << "]\n";
#endif /* DEBUG */
    if (i % 2 == 0) {
      assert(state_matrix[r][c] == NODE_STATE::PRIME);
      state_matrix[r][c] = NODE_STATE::STAR;
    } else {
      assert(state_matrix[r][c] == NODE_STATE::STAR);
      state_matrix[r][c] = NODE_STATE::NONE;
    }
  }

#if DEBUG
  show_state_matrix();
#endif /* DEBUG */
  return ALGO_STAGE::ONE;
}

int CVIMunkres::stage_4() {
#if DEBUG
  std::cout << "stage 4" << std::endl;
  show_state_matrix();
  std::cout << "cost matrix (before)" << std::endl;
  std::cout << cost_matrix << std::endl;
#endif /* DEBUG */
  float min = find_uncovered_min();

  for (int i = 0; i < m_size; i++) {
    for (int j = 0; j < m_size; j++) {
      if (covered_R[i] && covered_C[j]) {
        cost_matrix(i, j) += min;
      } else if (!covered_R[i] && !covered_C[j]) {
        cost_matrix(i, j) -= min;
      }
    }
  }

#if DEBUG
  std::cout << "cost matrix (after)" << std::endl;
  std::cout << cost_matrix << std::endl;
#endif /* DEBUG */

  return ALGO_STAGE::TWO;
}

int CVIMunkres::stage_final() {
#if DEBUG
  std::cout << "stage final\n";
  show_state_matrix();
  std::cout << "cost matrix\n";
  std::cout << cost_matrix << std::endl;
#endif /* DEBUG */

  match_result = new int[rows];
  for (int i = 0; i < rows; i++) {
    bool is_match = false;
    for (int j = 0; j < cols; j++) {
      if (state_matrix[i][j] == NODE_STATE::STAR) {
        match_result[i] = j;
        is_match = true;
        break;
      }
    }
    if (!is_match) {
      match_result[i] = -1;
    }
  }

  return ALGO_STAGE::DONE;
}

bool CVIMunkres::find_uncovered_zero(int &r, int &c) {
  for (int i = 0; i < m_size; i++) {
    if (covered_R[i]) {
      continue;
    }
    for (int j = 0; j < m_size; j++) {
      if (covered_C[j]) {
        continue;
      }
      if (cost_matrix(i, j) == 0) {
        r = i;
        c = j;
        return true;
      }
    }
  }
  return false;
}

float CVIMunkres::find_uncovered_min() {
  float min_value = __FLT_MAX__;
  for (int i = 0; i < m_size; i++) {
    if (covered_R[i]) {
      continue;
    }
    for (int j = 0; j < m_size; j++) {
      if (covered_C[j]) {
        continue;
      }
      if (cost_matrix(i, j) < min_value) {
        min_value = cost_matrix(i, j);
      }
    }
  }
#if DEBUG
  std::cout << "find uncovered min = " << min_value << std::endl;
#endif /* DEBUG */
  return min_value;
}

bool CVIMunkres::find_star_by_row(const int &row, int &c) {
  for (int j = 0; j < m_size; j++) {
    if (state_matrix[row][j] == NODE_STATE::STAR) {
      c = j;
      return true;
    }
  }
  return false;
}

bool CVIMunkres::find_prime_by_row(const int &row, int &c) {
  for (int j = 0; j < m_size; j++) {
    if (state_matrix[row][j] == NODE_STATE::PRIME) {
      c = j;
      return true;
    }
  }
  return false;
}

bool CVIMunkres::find_star_by_col(const int &col, int &r) {
  for (int i = 0; i < m_size; i++) {
    if (state_matrix[i][col] == NODE_STATE::STAR) {
      r = i;
      return true;
    }
  }
  return false;
}

// bool CVIMunkres::find_prime_by_col(int row, int &c){
//   /* TODO */
//   return false;
// }

void CVIMunkres::substract_min_by_row() {
  for (int i = 0; i < m_size; i++) {
    float min_value = cost_matrix(i, 0);
    for (int j = 1; j < m_size; j++) {
      if (min_value > cost_matrix(i, j)) {
        min_value = cost_matrix(i, j);
      }
    }

    for (int j = 0; j < m_size; j++) {
      cost_matrix(i, j) -= min_value;
    }
  }
}

void CVIMunkres::substract_min_by_col() {
  for (int j = 0; j < m_size; j++) {
    float min_value = cost_matrix(0, j);
    for (int i = 1; i < m_size; i++) {
      if (min_value > cost_matrix(i, j)) {
        min_value = cost_matrix(i, j);
      }
    }
    for (int i = 0; i < m_size; i++) {
      cost_matrix(i, j) -= min_value;
    }
  }
}

void CVIMunkres::show_state_matrix() {
  std::cout << "==============================================\n";
  std::cout << std::setw(8) << "STATE_M";
  for (int j = 0; j < m_size; j++) {
    std::cout << std::setw(8) << covered_C[j];
  }
  std::cout << std::endl;
  for (int i = 0; i < m_size; i++) {
    std::cout << std::setw(8) << covered_R[i];
    for (int j = 0; j < m_size; j++) {
      std::cout << std::setw(8) << state_matrix[i][j];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void CVIMunkres::show_result() {
  float total_cost = 0.0;
  for (int i = 0; i < rows; i++) {
    if (match_result[i] != -1) {
      int j = match_result[i];
      std::cout << "(" << std::setw(2) << i << "," << std::setw(2) << j << ") -> " << std::setw(8)
                << std::setprecision(4) << (*original_matrix)(i, j) << std::endl;
      total_cost += (*original_matrix)(i, j);
    }
  }

  std::cout << "total cost = " << std::setw(10) << std::setprecision(4) << total_cost << std::endl;
}