from collections import defaultdict


class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        ans = defaultdict(list)
        for item in strs:
            key = tuple(sorted(item))
            ans[key].append(item)
        return list(ans.values())