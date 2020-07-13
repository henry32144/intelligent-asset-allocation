import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import InputBase from '@material-ui/core/InputBase';
import IconButton from '@material-ui/core/IconButton';
import SearchIcon from '@material-ui/icons/Search';

const useStyles = makeStyles((theme) => ({
  root:{},
  stockComponent: {},
  searchBox: {
    margin: theme.spacing(2, 0, 0),
    padding: '2px 4px',
    display: 'flex',
    alignItems: 'center',
    minWidth: 330,
    width: '100%',
  },
  input: {
    marginLeft: theme.spacing(1),
    flex: 1,
  },
  iconButton: {
    padding: 10,
  },
  divider: {
    height: 28,
    margin: 4,
  },
}));
  
function StockSearchBox(props) {
  const { additionalStyles } = props
  const classes = useStyles();

  return (
    <div className={additionalStyles.stockComponent}>
      <Paper component="form" className={classes.searchBox}>
        <InputBase
          className={classes.input}
          placeholder="Search Stocks"
          inputProps={{ 'aria-label': 'search stocks' }}
        />
        <IconButton className={classes.iconButton} aria-label="search" onClick={(e) => {console.log("search on click")}}>
          <SearchIcon />
        </IconButton>
      </Paper>
    </div>
  );
}

export default StockSearchBox;