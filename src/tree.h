typedef enum {
    TREE_TYPE_INNER = 0,       // 0b00: both left and right are inners
    TREE_TYPE_LEFT_LEAF = 1,   // 0b01: left is a leaf
    TREE_TYPE_RIGHT_LEAF = 2,  // 0b10: right is a leaf // WILL NEVER OCCOUR
    TREE_TYPE_BOTH_LEAF = 3,   // 0b11: both are leaves
} tree_type_t;

typedef struct {
    tree_type_t t_type;
    double t_radius;
    void *t_left;   // tree_t or double
    void *t_right;  // tree_t or double
    double t_center[];
} tree_t;


inline bool tree_is_inner(tree_t const *t);
inline bool tree_has_left_leaf(tree_t const *t);
inline bool tree_has_right_leaf(tree_t const *t);

// consistent functions for assignind indices
inline ssize_t tree_left_node_idx(ssize_t parent);
inline ssize_t tree_right_node_idx(ssize_t parent);

inline size_t tree_sizeof();

inline tree_t *tree_index_to_ptr(tree_t const *tree_vec, ssize_t idx);
// Safety: this assumes both pointers are to the same contiguous array
//         and that base_ptr < ptr
inline ssize_t tree_ptr_to_index(tree_t const *base_ptr,
                                    tree_t const *ptr);

#ifndef PROFILE
void tree_print(tree_t const *tree_nodes, ssize_t max_idx,
                    ssize_t n_tree_nodes, ssize_t n_points);
#endif