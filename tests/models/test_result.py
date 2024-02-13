from signal_editor.models.result import ManualPeakEdits


class TestManualPeakEdits:
    # Creating a new instance of ManualPeakEdits should result in an object with empty added and removed lists.
    def test_new_instance(self):
        edits = ManualPeakEdits()
        assert edits.added == []
        assert edits.removed == []

    # Calling clear() on a ManualPeakEdits object should result in both added and removed lists being empty.
    def test_clear(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.removed = [4, 5, 6]
        edits.clear()
        assert edits.added == []
        assert edits.removed == []

    def test_add_single_integer_value_not_in_removed_list(self):
        edits = ManualPeakEdits()
        edits.removed = [1, 2, 3]
        edits.new_added(4)
        assert edits.added == [4]
        assert edits.removed == [1, 2, 3]

    # Adds a single integer value to the 'added' list if it is not already in the 'added' list
    def test_add_single_integer_value_not_in_added_list(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.new_added(4)
        assert edits.added == [1, 2, 3, 4]
        assert edits.removed == []

    # Adds a sequence of integer values to the 'added' list, removing any values that are in the 'removed' list
    def test_add_sequence_of_integer_values_removing_from_removed_list(self):
        edits = ManualPeakEdits()
        edits.removed = [1, 2, 3]
        edits.new_added([3, 4, 5])
        assert edits.added == [4, 5]
        assert edits.removed == [1, 2]

    # Adds a single integer value to the 'added' list even if it is already in the 'added' list
    def test_add_single_integer_value_already_in_added_list(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.new_added(2)
        assert edits.added == [1, 2, 3, 2]
        assert edits.removed == []

    # Adds a sequence of integer values to the 'added' list, even if some values are already in the 'added' list
    def test_add_sequence_of_integer_values_already_in_added_list(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.new_added([2, 3, 4])
        assert edits.added == [1, 2, 3, 2, 3, 4]
        assert edits.removed == []

    # Calling new_added() with a single integer argument that is not in the removed list should add the integer to the added list.
    def test_new_added_single_integer_not_in_removed(self):
        edits = ManualPeakEdits()
        edits.removed = [1, 2, 3]
        edits.new_added(4)
        assert edits.added == [4]
        assert edits.removed == [1, 2, 3]

    # Calling new_added() with a single integer argument that is in both added and removed lists should remove the integer from the removed list and not add it to the added list.
    def test_new_added_single_integer_in_both_lists(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.removed = [3, 4, 5]
        edits.new_added(3)
        assert edits.added == [1, 2, 3]
        assert edits.removed == [4, 5]

    # Calling new_removed() with a single integer argument that is in both added and removed lists should remove the integer from the added list and not add it to the removed list.
    def test_new_removed_single_integer_in_both_lists(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.removed = [3, 4, 5]
        edits.new_removed(3)
        assert edits.added == [1, 2]
        assert edits.removed == [3, 4, 5]

    # Calling new_added() with a sequence of integers that includes integers in both added and removed lists should remove those integers from the removed list and not add them to the added list.
    def test_new_added_sequence_integers_in_both_lists(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.removed = [3, 4, 5]
        edits.new_added([3, 4, 5])
        assert edits.added == [1, 2, 3]
        assert edits.removed == []

    # Calling new_added() with a single integer argument that is in the removed list should remove the integer from the removed list.
    def test_new_added_single_integer_in_removed_list(self):
        edits = ManualPeakEdits()
        edits.removed = [1, 2, 3]
        edits.new_added(2)
        assert edits.removed == [1, 3]

    # Calling new_added() with a sequence of integers should add all integers that are not in the removed list to the added list.
    def test_new_added_sequence_of_integers(self):
        edits = ManualPeakEdits()
        edits.removed = [1, 2, 3]
        edits.new_added([2, 4, 5])
        assert edits.added == [4, 5]

    # Calling new_removed() with a single integer argument that is not in the added list should add the integer to the removed list.
    def test_new_removed_single_integer_not_in_added_list(self):
        edits = ManualPeakEdits()
        edits.added = [1, 2, 3]
        edits.new_removed(4)
        assert edits.removed == [4]
