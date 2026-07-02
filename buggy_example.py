def get_average(numbers):
    return sum(numbers) / len(numbers)


def build_query(user_id):
    return "SELECT * FROM users WHERE id = " + user_id


def add_item(item, items=[]):
    items.append(item)
    return items
