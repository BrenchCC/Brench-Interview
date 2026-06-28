class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        map = {}
        for idx, num in enumerate(nums):
            map[num] = idx
        
        for i, num in enumerate(nums):
            j = map.get(target-num)
            if j is not None and i!=j:
                return [i, j]

