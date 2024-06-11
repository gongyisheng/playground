from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# LC 2
def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    p1 = l1
    p2 = l2
    dummyHead = ListNode(0)
    p3 = dummyHead
    needExtraOne = False
    while p1 or p2 or needExtraOne:
        v1 = p1.val if p1 else 0
        v2 = p2.val if p2 else 0
        _v = v1 + v2 + bool(needExtraOne)
        needExtraOne = _v >= 10
        node = ListNode(_v % 10)
        p3.next = node
        p3 = node
        p1 = p1.next if p1 else None
        p2 = p2.next if p2 else None
    return dummyHead.next
